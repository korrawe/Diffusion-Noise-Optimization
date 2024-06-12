import json
import os
import pickle
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.tensors import collate
from diffusion.gaussian_diffusion import GaussianDiffusion
from dno import DNO, DNOOptions
from model.cfg_sampler import ClassifierFreeSampleModel
from sample import dno_helper
from sample.condition import CondKeyLocationsLoss
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import (create_gaussian_diffusion,
                              create_model_and_diffusion, load_model_wo_clip)
from utils.output_util import (construct_template_variables, sample_to_motion,
                               save_multiple_samples)
from utils.parser_util import generate_args


def main(num_trials=3):
    num_ode_steps = 10
    OPTIMIZATION_STEP = 800
    #############################################
    ### Gradient Checkpointing
    # More DDIM steps will require more memory for full chain backprop.
    # Will need to use checkpointing for DDIM steps > 20
    gradient_checkpoint = False
    #############################################
    ### Task selection ###
    task = ""
    task = "trajectory_editing"
    # task = "dense_optimization"
    # task = "motion_projection"
    # task = "motion_blending"
    # task = "motion_inbetweening"

    print("##### Task: %s #####" % task)
    #############################################
    args = generate_args()
    args.device = 0
    # print(args.__dict__)
    # print(args.arch)
    args.use_ddim = True
    fixseed(args.seed)

    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")

    args.max_frames = 196
    fps = 20
    args.fps = fps
    n_frames = min(args.max_frames, int(args.motion_length * fps))
    motion_length_cut = 6.0
    gen_frames = int(motion_length_cut * fps)
    assert gen_frames <= n_frames, "gen_frames must be less than n_frames"
    args.gen_frames = gen_frames
    print("n_frames", n_frames)
    is_using_data = False
    skeleton = paramUtil.t2m_kinematic_chain

    dist_util.setup_dist(args.device)
    # Output directory
    if out_path == "":
        # out_path = os.path.join(os.path.dirname(args.model_path),
        #                         'samples_{}_{}_seed{}_{}'.format(name, niter, args.seed, time.strftime("%Y%m%d-%H%M%S")))
        out_path = os.path.join(
            os.path.dirname(args.model_path),
            "samples_{}_seed{}".format(niter, args.seed),
        )
        if args.text_prompt != "":
            out_path += "_" + args.text_prompt.replace(" ", "_").replace(".", "")

    out_path = os.path.join(out_path, task + "_dno")
    args.output_dir = out_path

    # This block must be called BEFORE the dataset is loaded
    assert args.text_prompt != "", "Please specify text_prompt"
    if args.text_prompt != "":
        texts = [args.text_prompt]
        args.num_samples = 1

    args.num_repetitions = 1
    args.batch_size = args.num_samples

    print("Loading dataset...")
    data = load_dataset(args, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, _ = create_model_and_diffusion(args, data)
    diffusion = create_gaussian_diffusion(args, timestep_respacing="ddim100")

    ###################################
    # LOADING THE MODEL FROM CHECKPOINT
    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(
            model
        )  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    model_device = next(model.parameters()).device
    ###################################
    collate_args = [
        {
            "inp": torch.zeros(n_frames),
            "tokens": None,
            "lengths": gen_frames,
        }
    ] * args.num_samples
    collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
    _, model_kwargs = collate(collate_args)

    # add CFG scale to batch
    if args.guidance_param != 1:
        model_kwargs["y"]["scale"] = (
            torch.ones(args.batch_size, device=dist_util.dev())
            * args.guidance_param
        )

    # Name for logging
    model_kwargs["y"]["log_name"] = out_path
    model_kwargs["y"]["traj_model"] = False
    #############################################

    all_motions = []
    all_lengths = []
    all_text = []
    obs_list = []
    kframes = []

    ### Prepare target for task ################
    gen_batch_size = 1
    args.gen_batch_size = gen_batch_size
    target = torch.zeros([gen_batch_size, args.max_frames, 22, 3], device=model_device)
    target_mask = torch.zeros_like(target, dtype=torch.bool)
    SHOW_TARGET_POSE = False

    ############################################
    # Output path
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)
    args_path = os.path.join(out_path, "args.json")
    with open(args_path, "w") as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)
    ############################################

    for rep_i in range(args.num_repetitions):
        assert args.num_repetitions == 1, "Not implemented"
        if args.load_from == '':
            # Run normal text-to-motion generation to get the starting motion
            sample = dno_helper.run_text_to_motion(
                args, diffusion, model, model_kwargs, data, n_frames
            )  # [1, 263, 1, 120]
            if task == "motion_blending":
                # Sample another motion to use it for blending example.
                model_kwargs["y"]["text"] = ["a person is jumping sideway to the right"]
                sample_2 = dno_helper.run_text_to_motion(
                    args, diffusion, model, model_kwargs, data, n_frames
                )
        else:
            # Load from file
            idx_to_load = 0
            load_from_x = os.path.join(args.load_from , "optimized_x.pt")
            sample = torch.load(load_from_x)[None, idx_to_load].clone()

        #######################
        ##### Edting here #####
        #######################

        # Visualize the generated motion
        # Take last (final) sample and cut to the target length
        gen_sample = sample[:, :, :, :gen_frames]
        # Convert motion representation to skeleton motion
        gen_motions, cur_lengths, initial_text = sample_to_motion(
            gen_sample,
            args,
            model_kwargs,
            model,
            gen_frames,
            data.dataset.t2m_dataset.inv_transform,
        )
        initial_motion = gen_motions[0][0].transpose(2, 0, 1)  # [120, 22, 3]

        task_info = {
            "task": task,
            "skeleton": skeleton,
            "initial_motion": initial_motion,
            "initial_text": initial_text,
            "device": model_device,
        }

        if task == "motion_blending":
            NUM_OFFSET = 20
            # No target around the seam
            SEAM_WIDTH = 10  # 15 # 10 # 5 # 3
            # Concat the two motions. Fill the later half until max_length with the second motion
            gen_sample = torch.cat(
                [
                    # sample[:, :, :, : gen_frames // 2],  # half from the first motion
                    sample[:, :, :, gen_frames // 2 :],  # half from the first motion
                    sample_2[
                        :, :, :, NUM_OFFSET : gen_frames // 2 + NUM_OFFSET
                    ],  # half from the second motion
                ],
                dim=-1,
            )  # This is for visualization of concat without blending
            gen_sample_full = torch.cat(
                [
                    sample[:, :, :, gen_frames // 2 :],  # half from the first motion
                    sample_2[
                        :,
                        :,
                        :,
                        NUM_OFFSET : gen_frames - (gen_frames // 2) + NUM_OFFSET,
                    ],  # half from the second motion
                ],
                dim=-1,
            )
            # Convert sample to motion
            combined_motions, cur_lengths, cur_texts = sample_to_motion(
                gen_sample_full,  # gen_sample,
                args,
                model_kwargs,
                model,
                gen_frames,
                data.dataset.t2m_dataset.inv_transform,
            )
            # combined_motions[0] shape [1, 22, 3, 196]
            combined_kps = (
                torch.from_numpy(combined_motions[0][0])
                .to(model_device)
                .permute(2, 0, 1)
            )  # [196, 22, 3]
            task_info["combine_motion"] = combined_kps

        #### Prepare everything for the task ####
        target, target_mask, kframes, is_noise_init, initial_motion, obs_list = (
            dno_helper.prepare_task(task_info, args)
        )

        #### Noise Optimization Config ####
        is_editing_task = not is_noise_init
        noise_opt_conf = DNOOptions(
            num_opt_steps=OPTIMIZATION_STEP, # 300 if is_editing_task else 500,
            diff_penalty_scale=2e-3 if is_editing_task else 0,
        )
        START_FROM_NOISE = is_noise_init

        if task == "motion_inbetweening":
            SHOW_TARGET_POSE = True

        # Repeat target to match num_trials
        if target.shape[0] == 1:  
            target = target.repeat(num_trials, 1, 1, 1)
            target_mask = target_mask.repeat(num_trials, 1, 1, 1)
        elif target.shape[0] != num_trials:
            raise ValueError("target shape is not 1 or equal to num_trials")

        # At this point, we need to have (1) target, (2) target_mask, (3) kframes, (4, optional) initial motion

        ########################################
        # TODO: move SMPL visualization to a separate function
        SAVE_FOR_RENDERING = False  # True
        if SAVE_FOR_RENDERING:
            # Save additional objects for rendering
            additional_objects = {
                "drag": [],
                "obs": [],
                "target": [],
            }
            # Save obstacles
            for obs_i, obs in enumerate(obs_list):
                additional_objects["obs"].append((obs[0][0], obs[0][1], obs[1]))
                # selected_index = [62, 90, 110]
                #         target_locations = [(0.5, 0.5), (1., 1.), (1.5, 1.5)]
            if task == "motion_inbetweening":
                for cur_kframe in kframes:
                    for joint_idx in range(22):
                        additional_objects["target"].append(
                            initial_motion[cur_kframe[0], joint_idx]
                        )

            elif task == "trajectory_editing":
                # selected_index = [70]

                # joint_idx = 0 # Pelvis
                joint_idx = 15  # Head
                # joint_idx = 21 # Right hand

                # Hack for hand and head editing
                Hack = True
                # Hack = False
                if Hack:
                    # selected_index = [70]
                    selected_index = [110]
                    target[:, :120, :, :] = (
                        torch.from_numpy(initial_motion)
                        .to(model_device)
                        .repeat(num_trials, 1, 1, 1)
                    )
                    # target_mask = target_mask.repeat(num_trials, 1, 1, 1)

                for idx, edit_index in enumerate(selected_index):
                    if joint_idx == 21:
                        change = 0.90  # 1.5
                        edit_from = initial_motion[edit_index, joint_idx, [0, 2, 1]]
                        edit_from = (edit_from[0], edit_from[1], edit_from[2])
                        edit_to = edit_from
                        edit_to = (edit_to[0], edit_to[1], edit_to[2] + change)
                        target_mask[:, :, :, :] = False
                        target_mask[:, edit_index, joint_idx, :] = True
                        target[:, edit_index, joint_idx, 1] = (
                            target[:, edit_index, joint_idx, 1] + change
                        )
                    elif joint_idx == 15:
                        change = -0.5
                        edit_from = initial_motion[edit_index, joint_idx, [0, 2, 1]]
                        edit_from = (edit_from[0], edit_from[1], edit_from[2])
                        edit_to = edit_from
                        edit_to = (edit_to[0], edit_to[1], edit_to[2] + change)
                        target_mask[:, :, :, :] = False
                        target_mask[:, edit_index, joint_idx, :] = True
                        target[:, edit_index, joint_idx, 1] = (
                            target[:, edit_index, joint_idx, 1] + change
                        )

                    elif joint_idx == 0:
                        edit_from = initial_motion[edit_index, 0, [0, 2]]
                        edit_from = (edit_from[0], edit_from[1], 0)
                        edit_to = target_locations[idx]
                        edit_to = (edit_to[0], edit_to[1], 0)
                    additional_objects["drag"].append((edit_from, edit_to))
            # Save target

            # Save additional_objects to a pickle file
            pickle_file = os.path.join(out_path, "additional_objects.pkl")
            with open(pickle_file, "wb") as f:
                pickle.dump(additional_objects, f)
        #########################################

        # optimize without text
        model_kwargs["y"]["text"] = [""]

        ######## DDIM inversion ########
        # Do inversion to get the initial noise for editing
        inverse_step = 999 # 99  # 30
        diffusion_invert = create_gaussian_diffusion(args, timestep_respacing="ddim1000")
        cur_t = inverse_step  # 100 - inverse_step
        dump_steps = [0, 5, 10, 20, 30, 40, 49]
        shape = (args.batch_size, model.njoints, model.nfeats, n_frames)

        # dump_steps = [0, 5, 10, 15, 20, 25, 29]
        if task == "motion_blending":
            # motion_to_invert = gen_sample_full.clone()
            motion_to_invert = sample.clone()
        else:
            motion_to_invert = sample.clone()
        inv_noise, pred_x0_list = ddim_invert(
            diffusion_invert,
            model,
            motion_to_invert,
            model_kwargs=model_kwargs,  # model_kwargs['y']['text'],
            dump_steps=dump_steps,
            num_inference_steps=inverse_step,
            clip_denoised=False,
        )

        # generate with a controlled number of steps to really see the quality.
        # goal: now what can the first row samples, the last row should be able to match.
        first_row_is_sampled_with_limited_quality = False

        if first_row_is_sampled_with_limited_quality:
            diffusion = create_gaussian_diffusion(
                args, timestep_respacing=f"ddim{num_ode_steps}"
            )
            dump_steps = [0, 1, 2, 3, 5, 7, 9]
            sample = diffusion.ddim_sample_loop(
                model,
                (args.batch_size, model.njoints, model.nfeats, n_frames),
                clip_denoised=not args.predict_xstart,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=True,
                dump_steps=dump_steps,
                noise=inv_noise,
                const_noise=False,
            )

        #######################################
        #######################################
        ## START OPTIMIZING
        #######################################
        #######################################

        opt_step = noise_opt_conf.num_opt_steps
        inter_out = []
        step_out_list = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.95]
        step_out_list = [int(aa * opt_step) for aa in step_out_list]
        step_out_list[-1] = opt_step - 1
        diffusion = create_gaussian_diffusion(args, f"ddim{num_ode_steps}")

        if START_FROM_NOISE:
            torch.manual_seed(0)
            # use the batch size that comes from main()
            gen_shape = [num_trials, model.njoints, model.nfeats, n_frames]
            cur_xt = torch.randn(gen_shape).to(model_device)
        else:
            cur_xt = inv_noise.detach().clone()
            cur_xt = cur_xt.repeat(num_trials, 1, 1, 1)

        cur_xt = cur_xt.detach().requires_grad_()

        loss_fn = CondKeyLocationsLoss(
            target=target,
            target_mask=target_mask,
            transform=data.dataset.t2m_dataset.transform_th,
            inv_transform=data.dataset.t2m_dataset.inv_transform_th,
            abs_3d=False,
            use_mse_loss=False,
            use_rand_projection=False,
            obs_list=obs_list,
        )
        criterion = lambda x: loss_fn(x, **model_kwargs)

        def solver(z):
            return ddim_loop_with_gradient(
                diffusion,
                model,
                (num_trials, model.njoints, model.nfeats, n_frames),
                model_kwargs=model_kwargs,
                noise=z,
                clip_denoised=False,
                gradient_checkpoint=gradient_checkpoint,
            )

        # start optimizing
        noise_opt = DNO(
            model=solver, criterion=criterion, start_z=cur_xt, conf=noise_opt_conf
        )
        out = noise_opt()

        for t in step_out_list:
            print("save optimize at", t)

            # aggregate the batch
            inter_step = []
            for i in range(num_trials):
                inter_step.append(out["hist"][i]["x"][t])
            inter_step = torch.stack(inter_step, dim=0)
            inter_out.append(inter_step)

        for i in range(num_trials):
            hist = out["hist"][i]
            # Plot loss
            for key in [
                "loss",
                "loss_diff",
                "loss_decorrelate",
                "grad_norm",
                "lr",
                "perturb_scale",
                "diff_norm",
            ]:
                plt.figure()
                if key in ["loss", "loss_diff", "loss_decorrelate"]:
                    plt.semilogy(hist["step"], hist[key])
                    # plt.ylim(top=0.4)
                    # Plot horizontal red line at lowest point of loss function
                    min_loss = min(hist[key])
                    plt.axhline(y=min_loss, color="r")
                    plt.text(0, min_loss, f"Min Loss: {min_loss:.4f}", color="r")
                else:
                    plt.plot(hist["step"], hist[key])
                plt.legend([key])
                plt.savefig(os.path.join(out_path, f"trial_{i}_{key}.png"))
                plt.close()

        final_out = out["x"].detach().clone()

        ### Concat generated motion for visualization
        if task == "motion_blending":
            motion_to_vis = torch.cat([sample, sample_2, gen_sample, final_out], dim=0)
            captions = [
                "Original 1",
                "Original 2",
                "Naive concatenation",
            ] + [f"Prediction {i+1}" for i in range(num_trials)]
            args.num_samples = 3 + num_trials

            # for ii in range(len(sample)):
            #     sample[ii][1] = sample_2[ii][0, :, :, :gen_frames]
            # # Plot half-half motion at the last row
            # sample[-1] = gen_sample[0]
        else:
            motion_to_vis = torch.cat([sample, final_out], dim=0)
            captions = [
                "Original",
                # "Reconstruction",
            ] + [f"Prediction {i+1}" for i in range(num_trials)]
            args.num_samples = 1 + num_trials

        # if task == "trajectory_editing":
        # save optimized sample
        torch.save(out["z"], os.path.join(out_path, "optimized_z.pt"))
        torch.save(out["x"], os.path.join(out_path, "optimized_x.pt"))

        ###################
        ##### save editing path #####
        # additional_objects = {
        #     "drag": ((0., 0., 0.), (2, 2., 0.)), # Start - end
        #     "obs": (1., -1., 0., 0.5), # x, y, z, radius
        #     "target": (0., 0., 0.), # x, y, z
        # }
        # # Save additional_objects to a pickle file
        # with open(pickle_file, "wb") as f:
        #     pickle.dump(additional_objects, f)

        num_dump_step = 1
        args.num_dump_step = num_dump_step

        # Convert sample to XYZ skeleton locations
        # Each return size [bs, 1, 3, 120]
        cur_motions, cur_lengths, cur_texts = sample_to_motion(
            motion_to_vis,  # sample,
            args,
            model_kwargs,
            model,
            gen_frames,
            data.dataset.t2m_dataset.inv_transform,
        )

        if task == "motion_projection":
            # Visualize noisy motion in the second row last column
            noisy_motion = (
                target[0, :gen_frames, :, :].detach().cpu().numpy().transpose(1, 2, 0)
            )
            noisy_motion = np.expand_dims(noisy_motion, 0)
            cur_motions[-1] = np.concatenate(
                [cur_motions[-1][0:1], noisy_motion, cur_motions[-1][1:]], axis=0
            )
            cur_lengths[-1] = np.append(cur_lengths[-1], cur_lengths[-1][0])
            cur_texts.append(cur_texts[0])

            captions = [
                "Original",
                "Noisy Motion",
            ] + [f"Prediction {i+1}" for i in range(num_trials)]
            args.num_samples = 2 + num_trials

        all_motions.extend(cur_motions)
        all_lengths.extend(cur_lengths)
        all_text.extend(cur_texts)

    ### Save videos
    total_num_samples = args.num_samples * args.num_repetitions * num_dump_step

    # After concat -> [r1_dstep_1, r2_dstep_1, r3_dstep_1, r1_dstep_2, r2_dstep_2, ....]
    all_motions = np.concatenate(all_motions, axis=0)  # [bs * num_dump_step, 1, 3, 120]
    all_motions = all_motions[
        :total_num_samples
    ]  # [bs, njoints, 3, seqlen]
    all_text = all_text[:total_num_samples]  # len() = args.num_samples * num_dump_step
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    npy_path = os.path.join(out_path, "results.npy")

    print(f"saving results file to [{npy_path}]")
    np.save(
        npy_path,
        {
            "motion": all_motions,
            "text": all_text,
            "lengths": all_lengths,
            "num_samples": args.num_samples,
            "num_repetitions": args.num_repetitions,
        },
    )

    with open(npy_path.replace(".npy", ".txt"), "w") as fw:
        fw.write("\n".join(all_text))
    with open(npy_path.replace(".npy", "_len.txt"), "w") as fw:
        fw.write("\n".join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.t2m_kinematic_chain

    sample_files = []
    num_samples_in_out_file = num_dump_step 
    (
        sample_print_template,
        row_print_template,
        all_print_template,
        sample_file_template,
        row_file_template,
        all_file_template,
    ) = construct_template_variables(args.unconstrained)

    for sample_i in range(args.num_samples):
        rep_files = []

        print("saving", sample_i)
        caption = all_text[sample_i]
        length = all_lengths[sample_i]
        motion = all_motions[sample_i].transpose(2, 0, 1)[:length]
        save_file = sample_file_template.format(0, sample_i)
        print(
            sample_print_template.format(caption, 0, sample_i, save_file)
        )
        animation_save_path = os.path.join(out_path, save_file)
        plot_3d_motion(
            animation_save_path,
            skeleton,
            motion,
            dataset=args.dataset,
            title=captions[sample_i],
            fps=fps,
            kframes=kframes,
            obs_list=obs_list,
            target_pose=target[0].cpu().numpy(),
            gt_frames=[kk for (kk, _) in kframes] if SHOW_TARGET_POSE else [],
        )
        rep_files.append(animation_save_path)

        # Check if we need to stack video
        sample_files = save_multiple_samples(
            args,
            out_path,
            row_print_template,
            all_print_template,
            row_file_template,
            all_file_template,
            caption,
            num_samples_in_out_file,
            rep_files,
            sample_files,
            sample_i,
        )

    abs_path = os.path.abspath(out_path)
    print(f"[Done] Results are at [{abs_path}]")


def load_dataset(args, n_frames):
    print(f"args: {args}")
    conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=args.max_frames,
        split="test",
        hml_mode="text_only",  # 'train'
        traject_only=False,
        # use_random_projection=args.use_random_proj,
        # random_projection_scale=args.random_proj_scale,
        # augment_type="none",
        # std_scale_shift=args.std_scale_shift,
        # drop_redundant=args.drop_redundant,
    )
    data = get_dataset_loader(conf)
    data.fixed_length = n_frames
    return data


def ddim_loop_with_gradient(
    diffusion: GaussianDiffusion,
    model,
    shape,
    noise=None,
    clip_denoised=False,
    denoised_fn=None,
    cond_fn=None,
    model_kwargs=None,
    device=None,
    progress=False,
    eta=0.0,
    gradient_checkpoint=False,
):
    if device is None:
        device = next(model.parameters()).device
    assert isinstance(shape, (tuple, list))
    if noise is not None:
        img = noise
    else:
        img = torch.randn(*shape, device=device)

    indices = list(range(diffusion.num_timesteps))[::-1]

    if progress:
        # Lazy import so that we don't depend on tqdm.
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    def grad_checkpoint_wrapper(func):
        def func_with_checkpoint(*args, **kwargs):
            return torch.utils.checkpoint.checkpoint(
                func, *args, **kwargs, use_reentrant=False
            )

        return func_with_checkpoint

    for i in indices:
        t = torch.tensor([i] * shape[0], device=device)
        sample_fn = diffusion.ddim_sample
        if gradient_checkpoint:
            sample_fn = grad_checkpoint_wrapper(sample_fn)
        out = sample_fn(
            model,
            img,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            eta=eta,
        )
        img = out["sample"]
    return img


@torch.no_grad()
def ddim_invert(
    diffusion: GaussianDiffusion,
    model,
    motion,  # image: torch.Tensor,
    model_kwargs,  # prompt,
    dump_steps=[],
    num_inference_steps=99,  # 100, # 50,
    eta=0.0,
    clip_denoised=False,
    **kwds,
):
    """
    invert a real motion into noise map with determinisc DDIM inversion
    """
    latents = motion
    print("latents shape: ", latents.shape)
    xt_list = [latents]
    pred_x0_list = [latents]
    indices = list(range(num_inference_steps))  # start_t #  - skip_timesteps))
    # from tqdm import tqdm

    for i, t in enumerate(tqdm(indices, desc="DDIM Inversion")):
        # print(i, t)
        t = torch.tensor([t] * latents.shape[0], device=latents.device)
        out = diffusion.ddim_reverse_sample(
            model,
            latents,
            t,
            model_kwargs=model_kwargs,
            eta=eta,
            clip_denoised=clip_denoised,
        )
        latents, pred_x0 = out["sample"], out["pred_xstart"]
        xt_list.append(latents)
        pred_x0_list.append(pred_x0)

    if len(dump_steps) > 0:
        pred_x0_list_out = []
        for ss in reversed(dump_steps):
            print("save step: ", ss)
            pred_x0_list_out.append(pred_x0_list[ss])
        # return latents, pred_x0_list_out
        return latents, pred_x0_list_out

    return latents


if __name__ == "__main__":
    main()
