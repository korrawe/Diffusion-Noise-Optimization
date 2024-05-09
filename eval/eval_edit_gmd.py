# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from dataclasses import asdict
from functools import partial
import glob
import math
from pprint import pprint
from utils.fixseed import fixseed
import os
import time
import numpy as np
import torch
import copy
import json
from tqdm import tqdm
from utils.parser_util import GenerateArgs, generate_args
from utils.model_util import (
    create_model_and_diffusion,
    load_saved_model,
    create_gaussian_diffusion,
)
from utils.output_util import (
    # sample_to_motion,
    construct_template_variables,
    save_multiple_samples,
)
from data_loaders.humanml.data.dataset import HumanML3D, abs3d_to_rel, sample_to_motion
from utils.generation_template import get_template
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from data_loaders.humanml.scripts.motion_process import process_file, recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders import humanml_utils
from data_loaders.humanml.utils.plot_script import plot_3d_motion, plot_3d_motion_static
import shutil
from data_loaders.tensors import collate
from sample.noise_optimizer import NoiseOptimizer, NoiseOptOptions
from data_loaders.humanml.utils.metrics import calculate_skating_ratio, compute_jitter

from torch.cuda import amp
from sample.condition import (
    CondKeyLocationsLoss,
    get_target_and_inpt_from_kframes_batch,
    CondKeyLocations
)
import os.path as osp

from sample.keyframe_pattern import get_kframes, get_obstacles

# For debugging
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
from multiprocessing import get_context
from functools import partial
from eval.calculate_fid import calculate_fid_given_two_populations


def plot_debug(motion_to_plot, name, gen_loader, length):
    plot_3d_motion(name, gen_loader.dataset.kinematic_chain, 
                   motion_to_plot[:length].detach().cpu().numpy(), 'length %d' % length, 'humanml', fps=20)


def calculate_results(motion_before_edit, generated_motions, target_motions, target_masks, 
                      max_frames, num_keyframe, text="", dataset: HumanML3D=None, 
                      holdout_before_edit_rep=None,
                      motion_before_edit_rep=None,
                      generated_motions_rep=None,
                      ):
    """
    Args:
        motion_before_edit: (tensor) [1, 196, 22, 3]
        generated_motions: (tensor) [num_samples, 196, 22, 3]
        target_masks: (tensor) [num_samples, 196, 22, 3]
        max_frames: (int)
        text: (string)
    """
    metrics = {
        "Foot skating": [],
        "Jitter": [],
        "Content preservation": [],
        "Objective Error": [],
    }
    metrics_before_edit = {
        "Foot skating": [],
        "Jitter": [],
        "Objective Error": [],
    }

    left_foot_id = 10
    right_foot_id = 11
    left_hand_id = 20
    right_hand_id = 21
    head_id = 15
    opt_batch_size = len(generated_motions) // len(motion_before_edit)
    bf_edit_content_list = []
    # Before edit
    for i in range(len(motion_before_edit)):
        before_edit_cut = motion_before_edit[i, :max_frames, :, :]
        skate_ratio, _ = calculate_skating_ratio(before_edit_cut.permute(1, 2, 0).unsqueeze(0)) # need input shape [bs, 22, 3, max_len]
        metrics_before_edit['Foot skating'].append(skate_ratio.item())
        metrics_before_edit['Jitter'].append(compute_jitter(before_edit_cut).item())
        for j in range(opt_batch_size):
            target_idx = i * opt_batch_size + j
            metrics_before_edit['Objective Error'].append((torch.norm((before_edit_cut - target_motions[target_idx])
                                                                       * target_masks[target_idx], dim=2).sum() / num_keyframe).item())
        if 'jumping' in text or 'jump' in text:
            before_edit_above_ground = (before_edit_cut[:, left_foot_id, 1] > 0.05) & (before_edit_cut[:, right_foot_id, 1] > 0.05)
            bf_edit_content_list.append(before_edit_above_ground)
        elif 'raised hands' in text:
            before_edit_above_head = ((before_edit_cut[:, left_hand_id, 1] > before_edit_cut[:, head_id, 1]) & 
                                            (before_edit_cut[:, right_hand_id, 1] > before_edit_cut[:, head_id, 1]))
            bf_edit_content_list.append(before_edit_above_head)
        elif 'crawling' in text:
            before_edit_head_below = (before_edit_cut[:, head_id, 1] < 1.50)
            bf_edit_content_list.append(before_edit_head_below)

    # fid
    def calculate_fid(gt_motion, holdout_motion, gen_motion):
        # assume that the length = max_length
        device = gt_motion.device
        gt_length = torch.tensor([max_frames] * len(gt_motion))
        holdout_length = torch.tensor([max_frames] * len(holdout_motion))
        gen_length = torch.tensor([max_frames] * len(gen_motion))

        # fid_gt_gt2 = calculate_fid_given_two_populations(gt_motion, holdout_motion, gt_length, holdout_length, dataset=dataset, 
        #                                     dataset_name='humanml', device=device, batch_size=64)
        # fid_gt_gen = calculate_fid_given_two_populations(gt_motion, gen_motion, gt_length, gen_length, dataset=dataset, 
        #                                     dataset_name='humanml', device=device, batch_size=64)

        holdout1, holdout2 = torch.chunk(holdout_motion, 2, dim=0)
        h1_length, h2_length = torch.chunk(holdout_length, 2, dim=0)
        fid_h1_h2 = calculate_fid_given_two_populations(holdout1, holdout2, h1_length, h2_length, dataset=dataset, 
                                            dataset_name='humanml', device=device, batch_size=64)
        fid_h1_gen = calculate_fid_given_two_populations(holdout1, gen_motion, h1_length, gen_length, dataset=dataset, 
                                            dataset_name='humanml', device=device, batch_size=64)
        return {
            # f"fid_gt_holdout{len(holdout_motion)}": fid_h1_h2,
            # "fid_gt_gen": fid_gt_gen,
            "fid_h1_h2": fid_h1_h2,
            "fid_h1_gen": fid_h1_gen,
        }

    for i in range(len(generated_motions)):
        # Generated
        gen_cut = generated_motions[i, :max_frames, :, :]
        skate_ratio, _ = calculate_skating_ratio(gen_cut.permute(1, 2, 0).unsqueeze(0))
        metrics['Foot skating'].append(skate_ratio.item())
        metrics['Jitter'].append(compute_jitter(gen_cut).item())
        metrics['Objective Error'].append((torch.norm((gen_cut - target_motions[i]) * target_masks[i], dim=2).sum() / num_keyframe).item())
        first_gen_idx = i // opt_batch_size
        # Compute content preservation
        if 'jumping' in text or 'jump' in text:
            # Compute the ratio of matched frames where the feet are above the ground or touching the ground
            # First compute which frames in the generated motion that the feet are above the ground
            gen_above_ground = (gen_cut[:, left_foot_id, 1] > 0.05) & (gen_cut[:, right_foot_id, 1] > 0.05)
            content_ratio = (gen_above_ground == bf_edit_content_list[first_gen_idx]).sum() / max_frames
        elif 'raised hands' in text:
            # Compute the ratio of matched frames where the hands are above the head
            gen_above_head = (gen_cut[:, left_hand_id, 1] > gen_cut[:, head_id, 1]) & (gen_cut[:, right_hand_id, 1] > gen_cut[:, head_id, 1])
            content_ratio = (gen_above_head == bf_edit_content_list[first_gen_idx]).sum() / max_frames
        elif 'crawling' in text:
            # Compute the ratio of matched frames where the head is below 1.5m
            gen_head_below = (gen_cut[:, head_id, 1] < 1.50)
            content_ratio = (gen_head_below == bf_edit_content_list[first_gen_idx]).sum() / max_frames
        else:
            content_ratio = 0
        metrics['Content preservation'].append(content_ratio.item())

    # Calculate FID
    if holdout_before_edit_rep is not None:
        assert motion_before_edit_rep is not None and generated_motions_rep is not None, f"motion_before_edit_rep and generated_motions_rep must be provided if holdout_before_edit_rep is provided"
        fid_dict = calculate_fid(motion_before_edit_rep, holdout_before_edit_rep, generated_motions_rep)
    else:
        fid_dict = {}
    return metrics, metrics_before_edit, fid_dict


def main():
    max_samples = 96 # 32 # 96
    opt_batch_size = 16 # 4
    num_total_batches = math.ceil(max_samples / opt_batch_size)
    # We will generate a new original motion for each batch
    n_keyframe = 1

    # noise_opt_conf = NoiseOptOptions(
    #     unroll_steps=10,  #
    #     opt_steps=opt_steps,  #
    #     optimizer="adam", # adam
    #     grad_mode='unit', # sign, unit
    #     lr=lr,
    #     perturb_scale=lr/100, # lr/100 best
    #     diff_penalty_scale=0,
    #     decorrelate_scale=2e6, # 2e6, 1e6 (for split grad), 100 (for combined grad),  1e3 (for unit grad, split)
    #     separate_backward_for_ode=True,
    #     standardize_z_after_step=True,
    #     postfix="_fixrandombias"
    # )
    opt_steps = 300 # 10
    lr = 5e-2 # 5e-2 (adam), 0.02 (sgd)
    #### Noise Optimization Config ####
    noise_opt_conf = NoiseOptOptions(
        unroll_steps=10, # 10,  #
        opt_steps=opt_steps,  #
        optimizer="adam", # adam
        grad_mode="unit",
        lr=lr,
        perturb_scale=0, # lr/100 best
        diff_penalty_scale=1e-2, #  0,
        # explode
        decorrelate_scale=0, # 1e3, # 1e3, # 1e6 (for split grad), 100 (for combined grad)
        separate_backward_for_ode=False,
        lr_warm_up_steps=50,
        lr_scheduler="cosine",
        lr_decay_steps=opt_steps,
    )
    args = generate_args()
    args.device = 0
    # args.use_ddim = True

    print(args.__dict__)
    print(args.arch)
    print("##### Additional Guidance Mode: %s #####" % args.guidance_mode)
    

    # Update args according to guidance mode
    args = get_template(args, template_name=args.guidance_mode)

    fixseed(args.seed)

    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")

    max_frames = 196
    fps = 20
    n_frames = max_frames
    cut_frames = max_frames
    print("n_frames", n_frames)
    dist_util.setup_dist(args.device)
    # Output directory
    # if out_path == "":
    out_path = os.path.join(
        os.path.dirname(args.model_path),
        "eval_edit_{}".format(niter),
    )
    out_path = os.path.join(out_path, f"seed{args.seed}")
    out_path += "_" + args.text_prompt.replace(" ", "_").replace(".", "")

    # out_path = os.path.join(out_path, noise_opt_conf.name)

    # this block must be called BEFORE the dataset is loaded
    if args.text_prompt != "":
        args.num_samples = num_total_batches
        texts = [args.text_prompt] * args.num_samples
    else:
        raise ValueError("Please specify either text_prompt or input_text")

    # NOTE: Currently not supporting multiple repetitions due to the way we handle trajectory model
    args.num_repetitions = 1

    assert (
        args.num_samples <= args.batch_size
    ), f"Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})"
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = (
        args.num_samples
    )  # Sampling a single batch from the testset, with exactly args.num_samples

    print("Loading dataset...")
    data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion_ori = create_model_and_diffusion(args, data)

    ###################################
    # LOADING THE MODEL FROM CHECKPOINT
    print(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path)  # , use_avg_model=args.gen_avg_model)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(
            model
        )  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    ###################################

    collate_args = [
        {
            "inp": torch.zeros(n_frames),
            "tokens": None,
            "lengths": cut_frames,
        }
    ] * args.num_samples

    is_t2m = any([args.input_text, args.text_prompt])
    if is_t2m:
        # t2m
        collate_args = [
            dict(arg, text=txt) for arg, txt in zip(collate_args, texts)
        ]

    _, model_kwargs = collate(collate_args)

    model_kwargs["y"]["traj_model"] = args.traj_only

    #############################################

    all_motions = []
    all_lengths = []
    all_text = []
    obs_list = []

    model_device = next(model.parameters()).device


    # Output path
    os.makedirs(out_path, exist_ok=True)
    args_path = os.path.join(out_path, "args.json")
    with open(args_path, "w") as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    ############################################

    # add CFG scale to batch
    if args.guidance_param != 1:
        model_kwargs["y"]["scale"] = (
            torch.ones(args.batch_size, device=dist_util.dev())
            * args.guidance_param
        )
    print("classifier scale", args.classifier_scale)
    #####################################################

    ### Evaluation ###
    # First generate initial motion with ddim 100
    # We will edit this motion to evaluate if the editing is successful 
    # and can it keep the content or the original motion

    # if args.use_ddim:
    #     sample_fn = diffusion_ori.ddim_sample_loop
    #     # dump_steps for logging progress
    #     dump_steps = [0, 1, 10, 30, 50, 70, 99]
    # else:
    #     sample_fn = diffusion_ori.p_sample_loop
    #     dump_steps = [999]

    # Pass functions to the diffusion
    diffusion_ori.data_get_mean_fn = data.dataset.t2m_dataset.get_std_mean
    diffusion_ori.data_transform_fn = data.dataset.t2m_dataset.transform_th
    diffusion_ori.data_inv_transform_fn = data.dataset.t2m_dataset.inv_transform_th

    ###################
    # MODEL INFERENCING
    ###################
    sample_file = os.path.join(out_path, "sample_before_edit.pt")
    # Load sample_before_edit from file if exists
    if os.path.exists(sample_file):
        print(f" - Sample already exists. Loading sample_before_edit from [{sample_file}]")
        sample = torch.load(sample_file)
    else:
        # List of samples of shape [bs, njoints, nfeats, nframes]
        # sample = diffusion_ori.ddim_sample_loop(
        #     model,
        #     (args.batch_size, model.njoints, model.nfeats, n_frames),
        #     # clip_denoised=False,
        #     clip_denoised=not args.predict_xstart,
        #     model_kwargs=model_kwargs,
        #     skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step # NOTE: testing this
        #     init_image=None,  # input_motions,  # init_image, # None, # NOTE: testing this
        #     progress=True,
        #     dump_steps=None,  # None,
        #     noise=None,
        #     const_noise=False,
        #     cond_fn=None,
        # )
        # # Save sample to file
        # torch.save(sample, sample_file)
        assert False, "This should not be called"

    sample_before_edit = sample.clone()
    #######################
    ##### Edting here #####
    #######################

    skeleton = (
        paramUtil.kit_kinematic_chain
        if args.dataset == "kit"
        else paramUtil.t2m_kinematic_chain
    )
    task = "trajectory_editing"
    # task = "dense_optimization"
    # task = "motion_projection"

    if task == "trajectory_editing":
        # Get obstacle list
        # obs_list = get_obstacles()

        ### Random sample the keyframes and target locations here ###
        obs_list = []
        # model_kwargs["y"]["text"] = ["a person who is standing with his arms held head high lifts his arms above his head, twice."]
        # selected_index = [9, 27, 46, 59, 62]
        # target_locations = [(0.0065, -0.0013), (0.0261, -0.0093), (0.0415, -0.0124), (-0.0191, -0.0572), (-0.0132, -0.0613)] # (-0.0132, -0.0613)]
        selected_index = [102]
        target_locations = [(2, 2)]
        target = torch.zeros([1, max_frames, 22, 3], device=model_device)
        target_mask = torch.zeros_like(target, dtype=torch.bool)
        kframes = [
            (tt, locs) for (tt, locs) in zip(selected_index, target_locations)
        ]
        for tt, locs in zip(selected_index, target_locations):
            # print("target at %d = %.1f, %.1f" % (tt, locs[0], locs[1]))
            target[0, tt, 0, [0, 2]] = torch.tensor(
                [locs[0], locs[1]], dtype=torch.float32, device=target.device
            )
            target_mask[0, tt, 0, [0, 2]] = True


    # # repeat num trials times on the first dimension 
    # target = target.repeat(opt_batch_size, 1, 1, 1)
    # target_mask = target_mask.repeat(opt_batch_size, 1, 1, 1)

    ######################
    ### DDIM INVERSION ###
    ######################
    # Set text to empty
    # model_kwargs["y"]["text"] = [""]

    # # dump_steps = [0, 5, 10, 15, 20, 25, 29]
    # inv_noise = diffusion_ori.invert(
    #     model,
    #     sample.clone(),
    #     model_kwargs=model_kwargs,
    #     dump_steps=[],
    #     num_inference_steps=99,
    # )


    ######################
    ## START OPTIMIZING ##
    ######################
    # Loop over target locations

    # target = target.repeat(opt_batch_size, 1, 1, 1)
    # target_mask = target_mask.repeat(opt_batch_size, 1, 1, 1)
    output_list = []
    target_list = []
    target_mask_list = []
    first_gen_list = []
    
    model_kwargs["y"]["text"] = model_kwargs["y"]["text"][0] * opt_batch_size
    if args.guidance_param != 1:
        model_kwargs["y"]["scale"] = (
            torch.ones(opt_batch_size, device=dist_util.dev())
            * args.guidance_param
        )
    
    for ii in range(num_total_batches):
        seed_number = ii
        fixseed(seed_number)

        ## Sample points    
        # reusing the target if it exists
        target_batch_file = f'target_{ii:04d}.pt'
        target_batch_file = os.path.join(out_path, target_batch_file)
        # if os.path.exists(target_batch_file):
        #     # [batch_size, n_keyframe]
        #     saved_target = torch.load(target_batch_file, map_location=model_device)
        #     target, target_mask = saved_target['target'], saved_target['target_mask']
        #     print(f'sample keyframes {target_batch_file} exists, loading from file')
        # else:
        min_frame = 60
        max_frame = 90
        sampled_keyframes =  ((max_frame - min_frame) * torch.rand(opt_batch_size, n_keyframe) + min_frame).long()
        max_x, max_z = 2.0, 2.0
        sampled_locations = (torch.rand(opt_batch_size, n_keyframe, 2) * 2 - 1) * torch.tensor([max_x, max_z])
        sampled_locations = sampled_locations.to(model_device)
        target = torch.zeros([opt_batch_size, max_frames, 22, 3], device=model_device)
        target_mask = torch.zeros_like(target, dtype=torch.bool)
        for bb in range(opt_batch_size):
            for jj in range(n_keyframe):
                target[bb, sampled_keyframes[bb, jj], 0, [0, 2]] = sampled_locations[bb, jj]
                target_mask[bb, sampled_keyframes[bb, jj], 0, [0, 2]] = True

        # torch.save({'target': target, 'target_mask': target_mask}, target_batch_file)

        # Prepare inpainting target
        # import pdb; pdb.set_trace()
        # 

        # kframes_num = [a for (a,b) in kframes] # [0, 30, 60, 90, 119]
        # kframes_posi = torch.tensor(kframes_num, dtype=torch.int).unsqueeze(0).repeat(opt_batch_size, 1)
        ### Prepare target
        # Get dummy skel_motions of shape [1, 22, 3, max_length] from keyframes
        # We do it this way so that get_target_...() can be shared across guidance modes.
        dummy_skel_motions = torch.zeros([opt_batch_size, 22, 3, n_frames])
        for jj in range(opt_batch_size):
            # dummy_skel_motions[jj, ]
            key_posi = target[jj, :, 0, :].detach().cpu()  # [max_length, 3]
            for kframe in sampled_keyframes[jj]:
                dummy_skel_motions[jj, 0, [0, 2], kframe] = key_posi[kframe, [0, 2]]
        # for (tt, locs) in kframes:
        #     print("target at %d = %.1f, %.1f" % (tt, locs[0], locs[1]))
        #     dummy_skel_motions[0, 0, [0, 2], tt] = torch.tensor([locs[0], locs[1]]).float()
        # dummy_skel_motions = dummy_skel_motions.repeat(opt_batch_size, 1, 1, 1)  # [1, 22, 3, max_length]

        (target_2, target_mask_2, 
         inpaint_traj, inpaint_traj_mask,
         inpaint_traj_points, inpaint_traj_mask_points,
         inpaint_motion, inpaint_mask, 
         inpaint_motion_points, inpaint_mask_points) = get_target_and_inpt_from_kframes_batch(dummy_skel_motions, sampled_keyframes, data.dataset)

        target = target.to(model_device)
        target_mask = target_mask.to(model_device)
        model_kwargs['y']['target'] = target
        model_kwargs['y']['target_mask'] = target_mask

        # import pdb; pdb.set_trace()


        # Add target to list
        for jj in range(opt_batch_size):
            target_list.append(target[jj])
            target_mask_list.append(target_mask[jj])

        batch_file = f"batch_{ii}.pt"
        batch_path = os.path.join(out_path, batch_file)
        # Load results if exists
        if os.path.exists(batch_path):
            print(f"Loading results from [{batch_path}]")
            final_motion = torch.load(batch_path)
        else:
            # Prepare target
            target = target
            target_mask = target_mask

            impute_slack = 20
            model_kwargs['y']['cond_until'] = impute_slack
            model_kwargs['y']['impute_until'] = impute_slack


            model_kwargs['y']['inpainted_motion'] = inpaint_motion.to(model_device) # init_motion.to(model_device)
            model_kwargs['y']['inpainting_mask'] = inpaint_mask.to(model_device)

            model_kwargs['y']['inpainted_motion_second_stage'] = inpaint_motion_points.to(model_device)
            model_kwargs['y']['inpainting_mask_second_stage'] = inpaint_mask_points.to(model_device)

            model_kwargs['y']['impute_until_second_stage'] = impute_slack


            cond_fn = CondKeyLocations(target=target,
                                        target_mask=target_mask,
                                        transform=data.dataset.t2m_dataset.transform_th,
                                        inv_transform=data.dataset.t2m_dataset.inv_transform_th,
                                        abs_3d=args.abs_3d,
                                        classifiler_scale=args.classifier_scale,
                                        use_mse_loss=False,
                                        use_rand_projection=args.use_random_proj
                                        )

            sample_fn_motion = diffusion_ori.p_sample_loop
            sample_motion = sample_fn_motion(
                            model,
                            (opt_batch_size, model.njoints, model.nfeats, n_frames),  # motion.shape
                            clip_denoised=not args.predict_xstart,
                            model_kwargs=model_kwargs,
                            skip_timesteps=0,
                            init_image=None,
                            progress=True,
                            dump_steps=None,
                            noise=None,
                            const_noise=False,
                            cond_fn=cond_fn
                        )
            # import pdb; pdb.set_trace()
            # save to file
            # torch.save(sample_motion, batch_path)
            final_motion = sample_motion

            # Save the results per batch
            torch.save(final_motion, batch_path)
        # Add to output list
        for jj in range(opt_batch_size):
            output_list.append(final_motion[jj])

    #######################
    ### COMPUTE RESULTS ###
    #######################

    generated_motions = []
    generated_motions_rep = []
    # convert the generated motion to skeleton
    for generated in output_list:
        generated_motions_rep.append(generated)
        generated = sample_to_motion(generated.unsqueeze(0), data.dataset, model, abs_3d=args.abs_3d)
        generated = generated.permute(0, 3, 1, 2)
        generated_motions.append(generated)
    # for FID
    generated_motions_rep = torch.stack(generated_motions_rep, dim=0)
    # NOTE: different from eval_edit.py
    # Load new dataset with rel setup so we can transform the motion to the original space
    args_cloned = copy.deepcopy(args)
    args_cloned.abs_3d = False
    args_cloned.use_random_proj = False
    data_rel = load_dataset(args_cloned, max_frames, n_frames)
    motion_before_edit = sample_to_motion(sample_before_edit, data_rel.dataset, model, abs_3d=False).permute(0, 3, 1, 2)

    # (num_samples, x, x, x)
    generated_motions = torch.cat(generated_motions, dim=0)
    target_motions = torch.stack(target_list, dim=0).detach().cpu()
    target_masks = torch.stack(target_mask_list, dim=0).detach().cpu()
    # import pdb; pdb.set_trace()
    # (num_samples, )
    
    # import pdb; pdb.set_trace()
    save_dir = out_path
    # save_dir = os.path.join(os.path.dirname(args.model_path), f'eval_{task}_{noise_opt_conf.name}')
    log_file = os.path.join(save_dir, f'eval_N{max_samples}.txt')

    DEBUG = True
    if DEBUG:
        print("Saving debug videos...")
        for ii in range(len(motion_before_edit)):
            before_edit_id = f'{ii:05d}'
            plot_debug(motion_before_edit[ii], osp.join(save_dir, f"before_edit_{before_edit_id}.mp4"), data, max_frames)

        start_from = 0 # 14
        for ii in range(start_from, len(generated_motions)): 
            motion_id = f'{ii:05d}'
            before_edit_id = f'{(ii//opt_batch_size):05d}'
            plot_debug(generated_motions[ii], osp.join(save_dir, f"{motion_id}_gen.mp4"), data, max_frames)
            # plot_debug(target_motions[ii], osp.join(save_dir, f"{motion_id}_target.mp4"), gen_loader, motion_lengths[ii])
            # Concat the two videos
            os.system(f"ffmpeg -y -loglevel warning -i {save_dir}/before_edit_{before_edit_id}.mp4 -i {save_dir}/{motion_id}_gen.mp4 -filter_complex hstack {save_dir}/{motion_id}_combined.mp4")
            # Remove the generated video
            os.system(f"rm {save_dir}/{motion_id}_gen.mp4")
            # if ii > 20:
            if ii > 5:
                break      
    
    do_calculate_fid = False
    sample_holdout = None
    metrics, metrics_before_edit, fid = calculate_results(motion_before_edit, generated_motions, target_motions, 
                                                     target_masks, max_frames, n_keyframe, text=args.text_prompt, 
                                                     dataset=data.dataset, 
                                                     motion_before_edit_rep=sample_before_edit if do_calculate_fid else None,
                                                     holdout_before_edit_rep=sample_holdout if do_calculate_fid else None,
                                                     generated_motions_rep=generated_motions_rep if do_calculate_fid else None,
                                                     )

    with open(log_file, 'w') as f:
        for (name, eval_results) in zip(["Before Edit", "After Edit"], [metrics_before_edit, metrics]):
            print(f"==================== {name} ====================")
            print(f"==================== {name} ====================", file=f, flush=True)
            for metric_name, metric_values in eval_results.items():
                metric_values = np.array(metric_values)
                unit_name = ""
                if metric_name == "Jitter":
                    unit_name = "(m/s^3)"
                elif metric_name == "Foot skating":
                    unit_name = "(ratio)"
                elif metric_name == "Content preservation":
                    unit_name = "(ratio)"
                elif metric_name == "Objective Error":
                    unit_name = "(m)"
                print(f"Metric [{metric_name} {unit_name}]: Mean {metric_values.mean():.4f}, Std {metric_values.std():.4f}")
                print(f"Metric [{metric_name} {unit_name}]: Mean {metric_values.mean():.4f}, Std {metric_values.std():.4f}", file=f, flush=True)
            
        # show fid
        print(f"==================== FID ====================")
        print(f"==================== FID ====================", file=f, flush=True)
        for k, v in fid.items():
            print(f"{k}: {v:.4f}")
            print(f"{k}: {v:.4f}", file=f, flush=True)

    return 

    # NOTE: hack; for the plotter to plot three rows.
    # args.num_samples = 3
    args.num_samples = 2 # + num_trials

    # Cut the generation to the desired length
    # NOTE: this is important for UNETs where the input must be specific size (e.g. 224)
    # but the output can be cut to any length
    gen_eff_len = min(sample[0].shape[-1], cut_frames)
    print("cut the motion length to", gen_eff_len)
    for j in range(len(sample)):
        sample[j] = sample[j][:, :, :, :gen_eff_len]
    ###################

    # num_dump_step = 1 # len(dump_steps)
    # args.num_dump_step = num_dump_step
    # # Convert sample to XYZ skeleton locations
    # # Each return size [bs, 1, 3, 120]
    # cur_motions, cur_lengths, cur_texts = sample_to_motion(
    #     sample,
    #     args,
    #     model_kwargs,
    #     model,
    #     gen_eff_len,
    #     data.dataset.t2m_dataset.inv_transform,
    # )
    if task == "motion_projection":
        # import pdb; pdb.set_trace()
        # Visualize noisy motion in the second row last column
        cur_motions[-1][1] = (
            target[0, :gen_eff_len, :, :].detach().cpu().numpy().transpose(1, 2, 0)
        )
    all_motions.extend(cur_motions)
    all_lengths.extend(cur_lengths)
    all_text.extend(cur_texts)

    ### Save videos
    total_num_samples = args.num_samples * args.num_repetitions * num_dump_step

    # After concat -> [r1_dstep_1, r2_dstep_1, r3_dstep_1, r1_dstep_2, r2_dstep_2, ....]
    all_motions = np.concatenate(all_motions, axis=0)  # [bs * num_dump_step, 1, 3, 120]
    all_motions = all_motions[
        :total_num_samples
    ]  # #       not sure? [bs, njoints, 6, seqlen]
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

    
    abs_path = os.path.abspath(out_path)
    print(f"[Done] Results are at [{abs_path}]")


def load_processed_file(model_device, batch_size, traject_only=False):
    """Load template file for trajectory imputing"""
    template_path = "./assets/template_joints.npy"
    init_joints = torch.from_numpy(np.load(template_path))
    from data_loaders.humanml.scripts.motion_process import (
        process_file,
        recover_root_rot_pos,
    )

    data, ground_positions, positions, l_velocity = process_file(
        init_joints.permute(0, 3, 1, 2)[0], 0.002
    )
    init_image = data
    # make it (1, 263, 1, 120)
    init_image = torch.from_numpy(init_image).unsqueeze(0).float()
    init_image = torch.cat([init_image, init_image[0:1, 118:119, :].clone()], dim=1)
    # Use transform_fn instead
    # init_image = (init_image - data.dataset.t2m_dataset.mean) / data.dataset.t2m_dataset.std
    init_image = init_image.unsqueeze(1).permute(0, 3, 1, 2)
    init_image = init_image.to(model_device)
    if traject_only:
        init_image = init_image[:, :4, :, :]

    init_image = init_image.repeat(batch_size, 1, 1, 1)
    return init_image, ground_positions


def load_dataset(args, max_frames, n_frames):
    conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=max_frames,
        split="test",
        hml_mode="text_only",  # 'train'
        use_abs3d=args.abs_3d,
        traject_only=args.traj_only,
        use_random_projection=args.use_random_proj,
        random_projection_scale=args.random_proj_scale,
        augment_type="none",
        std_scale_shift=args.std_scale_shift,
        drop_redundant=args.drop_redundant,
    )
    data = get_dataset_loader(conf)
    # what's this for?
    data.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
