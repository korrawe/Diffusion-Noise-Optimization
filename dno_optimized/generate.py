import os
import sys
from argparse import ArgumentParser
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard.writer import SummaryWriter

from data_loaders.get_data import DatasetConfig, get_dataset_loader
from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.tensors import collate
from dno_optimized.callbacks.save_top_k import SaveTopKCallback
from dno_optimized.options import GenerateOptions
from model.cfg_sampler import ClassifierFreeSampleModel
from sample import dno_helper
from sample.condition import CondKeyLocationsLoss
from sample.gen_dno import ddim_invert, ddim_loop_with_gradient
from utils import dist_util
from utils.callback_util import callbacks_from_options
from utils.dist_util import setup_dist
from utils.fixseed import fixseed
from utils.model_util import create_gaussian_diffusion, create_model_and_diffusion, load_model_wo_clip
from utils.output_util import construct_template_variables, sample_to_motion, save_multiple_samples

from .noise_optimizer import DNO, DNOOptions


def main(config_file: str, dot_list=None):
    if dot_list is None:
        dot_list = []
    # Create structured base config
    schema_with_defaults = OmegaConf.structured(GenerateOptions())
    # Merge with user args from file and dotlist
    user_args = OmegaConf.load(config_file)
    cli_args = OmegaConf.from_cli(dot_list)
    merged_config = OmegaConf.merge(schema_with_defaults, user_args, cli_args)
    # Convert back to plain dataclass so we can access computed properties
    args: GenerateOptions = OmegaConf.to_object(merged_config)  # type: ignore

    print("=========== OPTIONS ===========")
    print(OmegaConf.to_yaml(args))
    print("===============================")

    assert args.text_prompt != "", "Please specify text_prompt"
    assert args.gen_frames <= args.n_frames, "gen_frames must be less than n_frames"

    fixseed(args.seed)
    setup_dist(args.device)

    data, diffusion, model, model_device, model_kwargs, target = prepare_dataset_and_model(args)

    all_motions = []
    all_lengths = []
    all_text = []
    obs_list = []
    kframes = []
    show_target_pose = False

    assert args.num_repetitions == 1, "More repetitions are not yet implemented"
    # start for rep_i in range(args.num_repetitions):
    (
        start_from_noise,
        gen_sample,
        inv_noise,
        kframes,
        noise_opt_conf,
        obs_list,
        sample,
        sample_2,
        show_target_pose,
        target,
        target_mask,
    ) = prepare_optimization(
        args, data, diffusion, kframes, model, model_device, model_kwargs, obs_list, show_target_pose, target
    )

    #######################################
    #######################################
    ### OPTIMIZATION
    #######################################
    #######################################

    opt_step = noise_opt_conf.num_opt_steps
    inter_out = []
    step_out_list = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.95]
    step_out_list = [int(aa * opt_step) for aa in step_out_list]
    step_out_list[-1] = opt_step - 1
    diffusion = create_gaussian_diffusion(args, f"ddim{args.num_ode_steps}")

    if start_from_noise:
        torch.manual_seed(0)
        # use the batch size that comes from main()
        gen_shape = [args.num_trials, model.njoints, model.nfeats, args.n_frames]
        cur_xt = torch.randn(gen_shape).to(model_device)
    else:
        cur_xt = inv_noise.detach().clone()
        cur_xt = cur_xt.repeat(args.num_trials, 1, 1, 1)

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

    def criterion(x: torch.Tensor):
        return loss_fn(x, **model_kwargs)

    def solver(z):
        return ddim_loop_with_gradient(
            diffusion,
            model,
            (args.num_trials, model.njoints, model.nfeats, args.n_frames),
            model_kwargs=model_kwargs,
            noise=z,
            clip_denoised=False,
            gradient_checkpoint=args.gradient_checkpoint,
        )

    callbacks = callbacks_from_options(args)

    ######## Main optimization loop #######
    noise_opt = DNO(model=solver, criterion=criterion, start_z=cur_xt, conf=noise_opt_conf, callbacks=callbacks)
    out = noise_opt()
    #######################################

    # If there is a Top K callback, get its best model's state dict as output
    if topk := callbacks.get(SaveTopKCallback):
        out = topk.best_models[0].state_dict

    captions, cur_lengths, cur_motions, cur_texts, num_dump_step = process_results(
        args,
        data,
        gen_sample,
        inter_out,
        model,
        model_kwargs,
        out,
        sample,
        sample_2,
        step_out_list,
        target,
        out["stop_optimize"],
    )

    all_motions.extend(cur_motions)
    all_lengths.extend(cur_lengths)
    all_text.extend(cur_texts)
    # end for rep_i in range(args.num_repetitions):

    save_videos(
        all_lengths, all_motions, all_text, args, captions, kframes, num_dump_step, obs_list, show_target_pose, target
    )


def load_dataset(args, n_frames):
    pprint(f"args: {args}")
    conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=args.max_frames,
        split="test",
        hml_mode="text_only",  # 'train'
        traject_only=False,
        num_workers=args.dataloader_num_workers,
    )
    data = get_dataset_loader(conf)
    data.fixed_length = n_frames
    return data


def prepare_dataset_and_model(args):
    print("Loading dataset...")
    data = load_dataset(args, args.n_frames)

    print("Creating model and diffusion...")
    model, _ = create_model_and_diffusion(args, data)
    diffusion = create_gaussian_diffusion(args, timestep_respacing="ddim100")

    ###################################
    # LOADING THE MODEL FROM CHECKPOINT
    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)
    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    model_device = next(model.parameters()).device
    ###################################

    collate_args = [
        {"inp": torch.zeros(args.n_frames), "tokens": None, "lengths": args.gen_frames, "text": args.text_prompt}
    ]
    _, model_kwargs = collate(collate_args)

    # add CFG scale to batch
    if args.guidance_param != 1:
        model_kwargs["y"]["scale"] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

    ### Name for logging #######################
    model_kwargs["y"]["log_name"] = args.out_path
    model_kwargs["y"]["traj_model"] = False
    #############################################

    ### Prepare target for task ################
    gen_batch_size = 1
    args.gen_batch_size = gen_batch_size
    target = torch.zeros([gen_batch_size, args.max_frames, 22, 3], device=model_device)
    ############################################

    # Output path
    if args.out_path.exists():
        raise FileExistsError(args.out_path)
    args.out_path.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=args, f=args.out_path / "args.yml")
    ############################################

    return data, diffusion, model, model_device, model_kwargs, target


def prepare_optimization(
    args: GenerateOptions,
    data,
    diffusion,
    kframes,
    model,
    model_device,
    model_kwargs,
    obs_list,
    show_target_pose,
    target,
):
    sample_2 = None
    if args.load_from == "":
        # Run normal text-to-motion generation to get the starting motion
        sample = dno_helper.run_text_to_motion(
            args, diffusion, model, model_kwargs, data, args.n_frames
        )  # [1, 263, 1, 120]
        if args.task == "motion_blending":
            # Sample another motion to use it for blending example.
            model_kwargs["y"]["text"] = ["a person is jumping sideway to the right"]
            sample_2 = dno_helper.run_text_to_motion(args, diffusion, model, model_kwargs, data, args.n_frames)
    else:
        # Load from file
        idx_to_load = 0
        load_from_x = os.path.join(args.load_from, "optimized_x.pt")
        sample = torch.load(load_from_x)[None, idx_to_load].clone()

    ########################
    ##### Editing here #####
    ########################

    # Visualize the generated motion
    # Take last (final) sample and cut to the target length
    gen_sample = sample[:, :, :, : args.gen_frames]
    # Convert motion representation to skeleton motion
    gen_motions, cur_lengths, initial_text = sample_to_motion(
        gen_sample,
        args,
        model_kwargs,
        model,
        args.gen_frames,
        data.dataset.t2m_dataset.inv_transform,
    )
    initial_motion = gen_motions[0][0].transpose(2, 0, 1)  # [120, 22, 3]

    task_info = {
        "task": args.task,
        "skeleton": t2m_kinematic_chain,
        "initial_motion": initial_motion,
        "initial_text": initial_text,
        "device": model_device,
    }

    if args.task == "motion_blending":
        NUM_OFFSET = 20
        # No target around the seam
        # SEAM_WIDTH = 10  # 15 # 10 # 5 # 3
        # Concat the two motions. Fill the later half until max_length with the second motion
        gen_sample = torch.cat(
            [
                # sample[:, :, :, : args.gen_frames // 2],  # half from the first motion
                sample[:, :, :, args.gen_frames // 2 :],  # half from the first motion
                sample_2[:, :, :, NUM_OFFSET : args.gen_frames // 2 + NUM_OFFSET],  # half from the second motion
            ],
            dim=-1,
        )  # This is for visualization of concat without blending
        gen_sample_full = torch.cat(
            [
                sample[:, :, :, args.gen_frames // 2 :],  # half from the first motion
                sample_2[
                    :,
                    :,
                    :,
                    NUM_OFFSET : args.gen_frames - (args.gen_frames // 2) + NUM_OFFSET,
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
            args.gen_frames,
            data.dataset.t2m_dataset.inv_transform,
        )
        # combined_motions[0] shape [1, 22, 3, 196]
        combined_kps = torch.from_numpy(combined_motions[0][0]).to(model_device).permute(2, 0, 1)  # [196, 22, 3]
        task_info["combine_motion"] = combined_kps

    #### Prepare everything for the task ####
    target, target_mask, kframes, is_noise_init, initial_motion, obs_list = dno_helper.prepare_task(task_info, args)

    #### Noise Optimization Config ####
    is_editing_task = not is_noise_init
    noise_opt_conf: DNOOptions = args.dno
    noise_opt_conf.diff_penalty_scale = 2e-3 if is_editing_task else 0
    start_from_noise = is_noise_init

    if args.task == "motion_inbetweening":
        show_target_pose = True

    # Repeat target to match num_trials
    if target.shape[0] == 1:
        target = target.repeat(args.num_trials, 1, 1, 1)
        target_mask = target_mask.repeat(args.num_trials, 1, 1, 1)
    elif target.shape[0] != args.num_trials:
        raise ValueError("target shape is not 1 or equal to num_trials")

    # At this point, we need to have (1) target, (2) target_mask, (3) kframes, (4, optional) initial motion

    # Optimization is done without text
    model_kwargs["y"]["text"] = [""]

    ######## DDIM inversion ########
    # Do inversion to get the initial noise for editing
    inverse_step = 100  # 1000 for more previse inversion
    diffusion_invert = create_gaussian_diffusion(args, timestep_respacing=f"ddim{inverse_step}")
    dump_steps = [0, 5, 10, 20, 30, 40, 49]
    # dump_steps = [0, 5, 10, 15, 20, 25, 29]

    if args.task == "motion_blending":
        # motion_to_invert = gen_sample_full.clone()
        motion_to_invert = sample.clone()
    else:
        motion_to_invert = sample.clone()

    inv_noise, pred_x0_list = ddim_invert(
        diffusion_invert,
        model,
        motion_to_invert,
        model_kwargs=model_kwargs,
        dump_steps=dump_steps,
        num_inference_steps=inverse_step,
        clip_denoised=False,
    )

    # generate with a controlled number of steps to really see the quality.
    # goal: now what the first row samples, the last row should be able to match.
    first_row_is_sampled_with_limited_quality = False
    if first_row_is_sampled_with_limited_quality:
        diffusion = create_gaussian_diffusion(args, timestep_respacing=f"ddim{args.num_ode_steps}")
        dump_steps = [0, 1, 2, 3, 5, 7, 9]
        sample = diffusion.ddim_sample_loop(
            model,
            (args.batch_size, model.njoints, model.nfeats, args.n_frames),
            clip_denoised=not args.predict_xstart,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=dump_steps,
            noise=inv_noise,
            const_noise=False,
        )

    return (
        start_from_noise,
        gen_sample,
        inv_noise,
        kframes,
        noise_opt_conf,
        obs_list,
        sample,
        sample_2,
        show_target_pose,
        target,
        target_mask,
    )


def process_results(
    args: GenerateOptions,
    data,
    gen_sample,
    inter_out,
    model,
    model_kwargs,
    out,
    sample,
    sample_2,
    step_out_list,
    target,
    stop_optimize,
):
    # new: make here the list of the optimization steps which should be saved, since it is based on the fact if we did all optimization steps or not
    step_out_list = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.95]
    step_out_list = [int(aa * stop_optimize) for aa in step_out_list]
    step_out_list[-1] = stop_optimize - 1

    for t in step_out_list:
        print("save optimize at", t)

        # aggregate the batch
        inter_step = []
        for i in range(args.num_trials):
            inter_step.append(out["hist"][i]["x"][t])
        inter_step = torch.stack(inter_step, dim=0)
        inter_out.append(inter_step)

    for i in range(args.num_trials):
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
            plt.savefig(os.path.join(args.out_path, f"trial_{i}_{key}.png"))
            plt.close()

    final_out = out["x"].detach().clone()

    ### Concat generated motion for visualization
    if args.task == "motion_blending":
        motion_to_vis = torch.cat([sample, sample_2, gen_sample, final_out], dim=0)
        captions = [
            "Original 1",
            "Original 2",
            "Naive concatenation",
        ] + [f"Prediction {i + 1}" for i in range(args.num_trials)]
        args.num_samples = 3 + args.num_trials

    else:
        motion_to_vis = torch.cat([sample, final_out], dim=0)
        captions = [
            "Original",
        ] + [f"Prediction {i + 1}" for i in range(args.num_trials)]
        args.num_samples = 1 + args.num_trials

    torch.save(out["z"], os.path.join(args.out_path, "optimized_z.pt"))
    torch.save(out["x"], os.path.join(args.out_path, "optimized_x.pt"))

    ###################
    num_dump_step = 1
    args.num_dump_step = num_dump_step

    # Convert sample to XYZ skeleton locations
    # Each return size [bs, 1, 3, 120]
    cur_motions, cur_lengths, cur_texts = sample_to_motion(
        motion_to_vis,  # sample,
        args,
        model_kwargs,
        model,
        args.gen_frames,
        data.dataset.t2m_dataset.inv_transform,
    )

    if args.task == "motion_projection":
        # Visualize noisy motion in the second row last column
        noisy_motion = target[0, : args.gen_frames, :, :].detach().cpu().numpy().transpose(1, 2, 0)
        noisy_motion = np.expand_dims(noisy_motion, 0)
        cur_motions[-1] = np.concatenate([cur_motions[-1][0:1], noisy_motion, cur_motions[-1][1:]], axis=0)
        cur_lengths[-1] = np.append(cur_lengths[-1], cur_lengths[-1][0])
        cur_texts.append(cur_texts[0])

        captions = [
            "Original",
            "Noisy Motion",
        ] + [f"Prediction {i + 1}" for i in range(args.num_trials)]
        args.num_samples = 2 + args.num_trials

    return captions, cur_lengths, cur_motions, cur_texts, num_dump_step


def save_videos(
    all_lengths,
    all_motions,
    all_text,
    args,
    captions,
    kframes,
    num_dump_step,
    obs_list,
    show_target_pose,
    target,
    tb_writer: SummaryWriter | None = None,
):
    total_num_samples = args.num_samples * args.num_repetitions * num_dump_step

    # After concat -> [r1_dstep_1, r2_dstep_1, r3_dstep_1, r1_dstep_2, r2_dstep_2, ....]
    all_motions = np.concatenate(all_motions, axis=0)  # [bs * num_dump_step, 1, 3, 120]
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 3, seqlen]
    all_text = all_text[:total_num_samples]  # len() = args.num_samples * num_dump_step
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    npy_path = os.path.join(args.out_path, "results.npy")

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
        fw.write("\n".join([str(length) for length in all_lengths]))

    print(f"saving visualizations to [{args.out_path}]...")

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
        print(sample_print_template.format(caption, 0, sample_i, save_file))
        animation_save_path = os.path.join(args.out_path, save_file)
        plot_3d_motion(
            animation_save_path,
            t2m_kinematic_chain,
            motion,
            dataset=args.dataset,
            title=captions[sample_i],
            fps=args.fps,
            kframes=kframes,
            obs_list=obs_list,
            target_pose=target[0].cpu().numpy(),
            gt_frames=[kk for (kk, _) in kframes] if show_target_pose else [],
        )
        rep_files.append(animation_save_path)

        # Check if we need to stack video
        sample_files = save_multiple_samples(
            args,
            args.out_path,
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

    abs_path = os.path.abspath(args.out_path)
    print(f"[Done] Results are at [{abs_path}]")


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="python -m dno_optimized.generate",
        description="Edit a new motion that was generated by a text-to-motion model.",
    )
    parser.add_argument("config_file", type=str, help="Path to a YML config file.")
    parser.add_argument("dotlist", nargs="*", help="A dotlist of arguments parsed by omegaconf.")
    args = parser.parse_args()
    try:
        main(args.config_file, args.dotlist)
    except KeyboardInterrupt:
        print("Aborted.", file=sys.stderr)
        sys.exit(1)
