import os

import torch

from data_loaders.humanml.utils.plot_script import plot_3d_motion_static
from sample.keyframe_pattern import get_kframes, get_obstacles


def prepare_task(task_info, args):
    """At this point, we need to have (1) target, (2) target_mask, (3) kframes, 
    (4) whether to start from noise (is_noise_init), 
    (5, optional) initial motion, (6) obs_list
    If is_noise_init is True, the initial motion will not be used.
    return target, target_mask, kframes, is_noise_init, init_motion, obs_list
    """
    # Prepare empty target and empty mask based on the desire length.
    target = torch.zeros(
        [args.gen_batch_size, args.gen_frames, 22, 3], device=task_info["device"]
    )
    target_mask = torch.zeros_like(target, dtype=torch.bool)

    taskname = task_info["task"]
    if taskname  == "trajectory_editing":
        return task_trajectory_editing(task_info, args, target, target_mask)
    elif taskname == "pose_editing":
        raise NotImplementedError
    elif taskname == "dense_optimization":
        return task_dense_optimization(task_info, args, target, target_mask)
    elif taskname == "motion_projection":
        return task_motion_projection(task_info, args, target, target_mask)
    elif taskname == "motion_blending":
        return task_motion_blending(task_info, args, target, target_mask)
    elif taskname == "motion_inbetweening":
        return task_motion_inbetweening(task_info, args, target, target_mask)
    elif taskname == "happy_holidays":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown task name: {taskname}")
 
    # return target, target_mask, kframes, is_noise_init, init_motion, obs_list


def task_trajectory_editing(task_info, args, target, target_mask):
    """ Trajectory Editing task. The goal is, given an original motion and the editing target,
    we want to optimize the noise to be the one that can be mapped to the motions that satisfy
    the editing target.
    """
    # Get obstacle list
    if "use_obstacles" in args:
        obs_list = get_obstacles()
    else:
        obs_list = []
    USE_GUI = False # True
    if USE_GUI:
        out_path = args.output_dir
        animation_save_path = os.path.join(out_path, "before_edit.mp4")
        # For trajectory editing, obtain the frame indices and the target locations
        selected_index, target_locations = plot_3d_motion_static(
            animation_save_path,
            task_info["skeleton"],
            task_info["initial_motion"],
            dataset=args.dataset,
            title=task_info["initial_text"],
            fps=args.fps,
            traj_only=False,
            kframes=[],
            obs_list=obs_list,
            target_pose=None,
            gt_frames=[],
        )
    else:
        # selected_index = [62, 90, 110]  # [0] # 
        selected_index = [90]  # [0] # 
        # target_locations = [(0.5, 0.5), (1., 1.), (1.5, 1.5)] #  [(0,0)] # 
        target_locations = [(1.5, 1.5)]

    # Set up the new target based on the selected frames and the target locations
    kframes = [
        (tt, locs) for (tt, locs) in zip(selected_index, target_locations)
    ]
    for tt, locs in zip(selected_index, target_locations):
        print("target at %d = %.1f, %.1f" % (tt, locs[0], locs[1]))
        target[0, tt, 0, [0, 2]] = torch.tensor(
            [locs[0], locs[1]], dtype=torch.float32, device=target.device
        )
        target_mask[0, tt, 0, [0, 2]] = True
    
    is_noise_init = False
    return target, target_mask, kframes, is_noise_init, task_info["initial_motion"], obs_list


def task_pose_editing(task_info, args, target, target_mask):
    return target, target_mask, kframes, is_noise_init, init_motion, obs_list


def task_dense_optimization(task_info, args, target, target_mask):
    """Dense optimization. This task is only to test if noise optimization can reconstruct an arbitrary motion.
    The idea is, starting from random noise, we want to steer it toward a specific noise that can reconstruct
    the given motion.
    """
    is_noise_init = True
    kframes = []
    obs_list = []
    # Target is the generated motion
    init_motion_len = task_info["initial_motion"].shape[0]
    target[0, :init_motion_len, :, :] = torch.from_numpy(
        task_info["initial_motion"]).to(target.device)
    target_mask[0, :init_motion_len, :, :] = True
    return target, target_mask, kframes, is_noise_init, task_info["initial_motion"], obs_list


def task_motion_projection(task_info, args, target, target_mask):
    """Motion projection (same as motion denoising). Given a set of noisy joints, we want to reconstruct
    a valid motion that is as close as possible to the noisy input.
    Start from random noise and use the noisy input as target. Functionally, this is the same as 
    the dense optimization task about.
    """
    is_noise_init = True
    kframes = []
    obs_list = []
    # Target is the generated motion
    init_motion_len = task_info["initial_motion"].shape[0]
    target[0, :init_motion_len, :, :] = torch.from_numpy(
        task_info["initial_motion"]).to(target.device)
    target_mask[0, :init_motion_len, :, :] = True

    ###### Add noise to target  ######
    # To ensure that there is a valid solution that can be reconstructed from the given noise,
    # we construct the noisy target by adding noise the given motion.
    noise_level = 0.03  # 0.01
    target = target + (torch.randn_like(target)) * noise_level

    return target, target_mask, kframes, is_noise_init, task_info["initial_motion"], obs_list


def task_motion_inbetweening(task_info, args, target, target_mask):
    """Motion in-betweening. Select two frames from a given motion to be use a starting frame and
    ending frame. Then, infill the motion in-between these two poses.
    """
    is_noise_init = True
    kframes = []
    obs_list = []
    init_motion_len = task_info["initial_motion"].shape[0]
    target[0, :init_motion_len, :, :] = torch.from_numpy(
        task_info["initial_motion"]).to(target.device)

    # Select two frames to be used as starting and ending frames.
    start_frame = 0
    target_frame = 80

    target_mask[0, [start_frame, target_frame], :, :] = True
    kframes = [(start_frame, (0,0)), (target_frame, (0,0))]

    return target, target_mask, kframes, is_noise_init, task_info["initial_motion"], obs_list


def task_motion_blending(task_info, args, target, target_mask):
    """Motion blending. Concat two initial motions together. 
    To create target, combine the motion in the representation space such that
    when we concat it, the second motion will start where the first motion ends.
    """
    # is_noise_init = False
    is_noise_init = True
    kframes = []
    obs_list = []

    # No target around the seam
    SEAM_WIDTH = 10 # 15 # 10 # 5 # 3

    # combined_motions[0] shape [1, 22, 3, 196]
    target[0] = task_info["combine_motion"]
    target_mask = torch.ones_like(target, dtype=torch.bool)
    target_mask[0, :, :, :] = True
    target_mask[0, args.gen_frames // 2 - SEAM_WIDTH: args.gen_frames // 2 + SEAM_WIDTH] = False

    return target, target_mask, kframes, is_noise_init, task_info["initial_motion"], obs_list


def run_text_to_motion(args, diffusion, model, model_kwargs, data, init_motion_length):
    if args.use_ddim:
        sample_fn = diffusion.ddim_sample_loop
        # # dump_steps for logging progress
        # dump_steps = [0, 1, 10, 30, 50, 70, 99]
    else:
        sample_fn = diffusion.p_sample_loop
        # # dump_steps = [1, 100, 300, 500, 700, 850, 999]
        # dump_steps = [999]

    # Pass functions to the diffusion
    diffusion.data_get_mean_fn = data.dataset.t2m_dataset.get_std_mean
    diffusion.data_transform_fn = data.dataset.t2m_dataset.transform_th
    diffusion.data_inv_transform_fn = data.dataset.t2m_dataset.inv_transform_th

    ###################
    # MODEL INFERENCING
    ###################

    # list of [bs, njoints, nfeats, nframes] each element is a different time step
    sample = sample_fn(
        model,
        (args.batch_size, model.njoints, model.nfeats, init_motion_length),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,
        init_image=None,
        progress=True,
        # dump_steps=dump_steps,
        noise=None,
        const_noise=False,
        cond_fn=None,
    )
    return sample