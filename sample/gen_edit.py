# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from dataclasses import asdict
from functools import partial
from pprint import pprint
from utils.fixseed import fixseed
import os
import time
import numpy as np
import torch
import copy
import json
from tqdm import tqdm
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_saved_model, create_gaussian_diffusion
from utils.output_util import sample_to_motion, construct_template_variables, save_multiple_samples
from utils.generation_template import get_template
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders import humanml_utils
from data_loaders.humanml.utils.plot_script import plot_3d_motion, plot_3d_motion_static
import shutil
from data_loaders.tensors import collate
# import flag
from torch.cuda import amp
from sample.condition import (get_target_from_kframes, get_inpainting_motion_from_traj, 
                              get_target_and_inpt_from_kframes_batch, 
                              cond_fn_key_location, cond_fn_sdf, log_trajectory_from_xstart,
                              CondKeyLocations, CondKeyLocationsWithSdf,
                              CondKeyLocationsLoss) # Testing SDS loss

from sample.keyframe_pattern import get_kframes, get_obstacles
# For debugging
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns


def load_reward_model(data):
    '''
    Create a reward model to help computing grad_{x_t} for traj conditioning.
    '''
    args_reward = generate_args(trajectory_model=True)  #
    args_reward.model_path = "./save/my_traj/model000400000.pt"
    args_reward.predict_xstart = True
    args_reward.abs_3d = True
    args_reward.traj_only = True

    reward_model, _ = create_model_and_diffusion(args_reward, data)
    print(
        f"Loading reward model checkpoints from [{args_reward.model_path}]..."
    )
    load_saved_model(reward_model, args_reward.model_path)  # , use_avg_model=args_reward.gen_avg_model)

    if args_reward.guidance_param != 1:
        reward_model = ClassifierFreeSampleModel(
            reward_model
        )  # wrapping model with the classifier-free sampler
    reward_model.to(dist_util.dev())
    reward_model.eval()  # disable random masking
    return reward_model


def load_traj_model(data):
    '''
    The trajectory model predicts trajectory that will be use for infilling in motion model.
    Create a trajectory model that produces trajectory to be inptained by the motion model.
    '''
    print("Setting traj model ...")
    # NOTE: Hard-coded trajectory model location
    traj_model_path = "./save/traj_unet_adazero_swxs_eps_abs_fp16_clipwd_224/model000062500.pt"
    args_traj = generate_args(model_path=traj_model_path)

    # print(args_traj.__dict__)
    # print(args_traj.arch)
    traj_model, traj_diffusion = create_model_and_diffusion(args_traj, data)

    print(f"Loading traj model checkpoints from [{args_traj.model_path}]...")
    load_saved_model(traj_model, args_traj.model_path)

    if args_traj.guidance_param != 1:
        traj_model = ClassifierFreeSampleModel(
            traj_model)  # wrapping model with the classifier-free sampler
    traj_model.to(dist_util.dev())
    traj_model.eval()  # disable random masking
    return traj_model, traj_diffusion


def main():
    args = generate_args()
    print(args.__dict__)
    print(args.arch)
    print("##### Additional Guidance Mode: %s #####" % args.guidance_mode)


    # import doodl.doodl as doodl
    # import pdb; pdb.set_trace()
    args.use_ddim = True



    # Update args according to guidance mode
    # args = get_template(args, template_name="testing")
    # args = get_template(args, template_name="kps") # mdm_legacy # trajectory # guidance_mode
    args = get_template(args, template_name=args.guidance_mode)

    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')

    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = min(max_frames, int(args.motion_length * fps))
    cut_frames = int(args.motion_length_cut * fps)
    print('n_frames', n_frames)
    is_using_data = not any([
        args.input_text, args.text_prompt, args.action_file, args.action_name
    ])
    dist_util.setup_dist(args.device)
    # Output directory
    if out_path == '':
        # out_path = os.path.join(os.path.dirname(args.model_path),
        #                         'samples_{}_{}_seed{}_{}'.format(name, niter, args.seed, time.strftime("%Y%m%d-%H%M%S")))
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_seed{}'.format(niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace(
                '.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace(
                '.txt', '').replace(' ', '_').replace('.', '')

    # this block must be called BEFORE the dataset is loaded
    if args.text_prompt != '':
        texts = [args.text_prompt]
        # args.num_samples = 1
        # Do 3 repetitions from the same propmt. But put it in num_sample instead so we can do all of them in parallel
        # NOTE: change this to 1 for editing
        args.num_samples = 1 #  3
        args.num_repetitions = 1
    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        args.num_samples = len(texts)
    elif args.action_name:
        action_text = [args.action_name]
        args.num_samples = 1
    elif args.action_file != '':
        assert os.path.exists(args.action_file)
        with open(args.action_file, 'r') as fr:
            action_text = fr.readlines()
        action_text = [s.replace('\n', '') for s in action_text]
        args.num_samples = len(action_text)
    
    # NOTE: Currently not supporting multiple repetitions due to the way we handle trajectory model
    args.num_repetitions = 1

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    ###################################
    # LOADING THE MODEL FROM CHECKPOINT
    print(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path) # , use_avg_model=args.gen_avg_model)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(
            model)  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    ###################################

    if is_using_data:
        iterator = iter(data)
        _, model_kwargs = next(iterator)
    else:
        collate_args = [{
            'inp': torch.zeros(n_frames),
            'tokens': None,
            # this would be incorrect for UNET models
            # 'lengths': n_frames,
            'lengths': cut_frames,
        }] * args.num_samples
        # model_kwargs['y']['lengths']
        is_t2m = any([args.input_text, args.text_prompt])
        if is_t2m:
            # t2m
            collate_args = [
                dict(arg, text=txt) for arg, txt in zip(collate_args, texts)
            ]
        else:
            # a2m
            action = data.dataset.action_name_to_action(action_text)
            collate_args = [
                dict(arg, action=one_action, action_text=one_action_text)
                for arg, one_action, one_action_text in zip(
                    collate_args, action, action_text)
            ]
        
        _, model_kwargs = collate(collate_args)
    
    # Name for logging
    model_kwargs['y']['log_name'] = out_path
    model_kwargs['y']['traj_model'] = args.traj_only
    # TODO: move two-staged model to a new class
    #########################################
    # Load another model for reward function
    if args.gen_reward_model:
        reward_model = load_reward_model(data)
        reward_model_kwargs = copy.deepcopy(model_kwargs)
    #########################################
    # loading another model for trajectory conditioning
    if args.gen_two_stages:
        traj_model, traj_diffusion = load_traj_model(data)
        traj_model_kwargs = copy.deepcopy(model_kwargs)
        traj_model_kwargs['y']['log_name'] = out_path
        traj_model_kwargs['y']['traj_model'] = True
    #############################################

    all_motions = []
    all_lengths = []
    all_text = []
    obs_list = []

    # NOTE: test for classifier-free sampling
    USE_CLASSIFIER_FREE = False # True

    model_device = next(model.parameters()).device
    # Load preprocessed file for inpainting test
    # [3, 263, 1, 120]
    input_motions, ground_positions = load_processed_file(model_device, args.batch_size, args.traj_only)
    input_skels = recover_from_ric(input_motions.permute(0, 2, 3, 1), 22, abs_3d=False)
    input_skels = input_skels.squeeze(1)
    # input_skels = input_skels[0].transpose(0, 3, 1, 2)
    # Get key frames for guidance
    if args.guidance_mode == "trajectory" or args.guidance_mode == "mdm_legacy":
        # Get key frames for guidance
        kframes = get_kframes(ground_positions) # ground_positions=ground_positions)
        # model_kwargs['y']['kframes_pattern'] = kframes
    elif args.guidance_mode == "kps":
        if args.interactive:
            # Get key frames for guidance from interactive GUI
            import pdb; pdb.set_trace()
            kframes = ()
        else:
            kframes = get_kframes(pattern="zigzag")
        model_kwargs['y']['kframes_pattern'] = kframes
    elif args.guidance_mode == "sdf":
        kframes = get_kframes(pattern="sdf")
        model_kwargs['y']['kframes_pattern'] = kframes
        obs_list = get_obstacles()
    elif USE_CLASSIFIER_FREE:
        kframes = get_kframes(pattern="zigzag", interpolate=True)

        # kframes = []
        # kframes = get_kframes(ground_positions) # ground_positions=ground_positions)
        model_kwargs['y']['kframes_pattern'] = kframes
    else:
        kframes = []

    # NOTE: test SDS
    TEST_SDS = False
    if TEST_SDS:
        kframes = get_kframes(pattern="3dots") # zigzag

    # TODO: remove mdm_legacy
    if args.guidance_mode == "mdm_legacy" and args.do_inpaint:
        # When use MDM (or relative model) and inpainting, be sure to set imputation mode
        # to "IMPUTE_AT_X0 = False" in gaussian_diffusion.py
        model_kwargs['y']['impute_relative'] = True
        model_kwargs['y']['inpainted_motion'] = input_motions.clone()
        # Get impainting mask
        inpainting_mask = torch.tensor(
            humanml_utils.HML_ROOT_MASK,
            dtype=torch.bool,
            device=model_device)  # True is root (global location)
        # Do not need to fix y
        inpainting_mask[3:] = False
        # model_kwargs['y']['inpainting_mask'][0] = False
        inpainting_mask = inpainting_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(
                input_motions.shape[0], 1, input_motions.shape[2],
                input_motions.shape[3])
        model_kwargs['y']['inpainting_mask'] = inpainting_mask
        
        motion_cond_until = 0
        motion_impute_until = 0
    else:
        motion_cond_until = 20
        motion_impute_until = 1
        
    #### Standardized conditioning
    # TODO: clean this for each guidance mode
    # Use the same function call as used during evaluation (condition.py)
    kframes_num = [a for (a,b) in kframes] # [0, 30, 60, 90, 119]
    kframes_posi = torch.tensor(kframes_num, dtype=torch.int).unsqueeze(0).repeat(args.batch_size, 1)

    ### Prepare target
    # Get dummy skel_motions of shape [1, 22, 3, max_length] from keyframes
    # We do it this way so that get_target_...() can be shared across guidance modes.
    dummy_skel_motions = torch.zeros([1, 22, 3, n_frames])
    for (tt, locs) in kframes:
        print("target at %d = %.1f, %.1f" % (tt, locs[0], locs[1]))
        dummy_skel_motions[0, 0, [0, 2], tt] = torch.tensor([locs[0], locs[1]])
    dummy_skel_motions = dummy_skel_motions.repeat(args.batch_size, 1, 1, 1)  # [1, 22, 3, max_length]

    (target, target_mask, 
        inpaint_traj_p2p, inpaint_traj_mask_p2p,
        inpaint_traj_points, inpaint_traj_mask_points,
        inpaint_motion_p2p, inpaint_mask_p2p,
        inpaint_motion_points, inpaint_mask_points) = get_target_and_inpt_from_kframes_batch(dummy_skel_motions, kframes_posi, data.dataset)
    target = target.to(model_device)
    target_mask = target_mask.to(model_device)
    model_kwargs['y']['target'] = target
    model_kwargs['y']['target_mask'] = target_mask

    ### NOTE: Prepare target for pose editing #######################
    POSE_EDITING = True # False
    if POSE_EDITING:
        batch_size = 1
        max_length = 196
        target_device = model_device
        target = torch.zeros([batch_size, max_length, 22, 3], device=target_device)
        target_mask = torch.zeros_like(target, dtype=torch.bool)

        ### For loss target, we only compute with respect to the key points
        # for (kframe, posi) in kframes:
        kframes = [(0, (0.0, 0.0)), (80, (0.0, 0.0))]
        for kframe in [0, 80]:
            target[0, kframe, :, :] = input_skels[0, kframe, :, :]
            target_mask[0, kframe, :, :] = True
        
    ###########################################
    # NOTE: Test imputing with poses
    # TODO: Delete this
    GUIDE_WITH_POSES = False
    # GUIDE_WITH_POSES = True
    if GUIDE_WITH_POSES:
        print("Guide with poses")
        # Get impainting mask
        inpainting_mask = torch.tensor(
            np.array([False] + [True]*2 + [False] + [True]*63 + [False]*(263-67)),
            dtype=torch.bool,
            device=model_device)  # [True] is root (global location)
        # model_kwargs['y']['inpainting_mask'][0] = False
        inpainting_mask = inpainting_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(
                input_motions.shape[0], 1, input_motions.shape[2],
                n_frames)
        model_kwargs['y']['inpainting_mask'] = inpainting_mask
        inpaint_mask_points = inpainting_mask
        input_motions[:, [1, 2], 0, :] = torch.from_numpy(ground_positions[:, 0, [0, 2]]).to(input_motions.device).permute(1, 0).unsqueeze(0).repeat(3, 1, 1)
        inpaint_motion_points = torch.cat([input_motions, 
                                        torch.zeros(*input_motions.shape[:3], n_frames - input_motions.shape[3], device=input_motions.device)], dim=3)
        for i in range(inpaint_mask_points.shape[-1]):
            if i not in [kk for (kk, _) in kframes]:
                inpaint_mask_points[:, :, :, i] = False
    ###########################################

    # Output path
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)
    args_path = os.path.join(out_path, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    ############################################
    # Generate trajectory
    # NOTE: num_repetitions > 1 is currently not supported
    for rep_i in range(args.num_repetitions):
        assert args.num_repetitions == 1, "Not implemented"

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(
                args.batch_size, device=dist_util.dev()) * args.guidance_param
            if args.gen_reward_model:
                reward_model_kwargs['y']['scale'] = torch.ones(
                    args.batch_size,
                    device=dist_util.dev()) * args.guidance_param
            if args.gen_two_stages:
                traj_model_kwargs['y']['scale'] = torch.ones(
                    args.batch_size,
                    device=dist_util.dev()) * args.guidance_param

        print("classifier scale", args.classifier_scale)

        # Standardized conditioning
        impute_slack = 20
        impute_until = 100

        #####################################################
        # If using TWO_STAGES, generate the trajectory first
        if args.gen_two_stages:
            traj_model_kwargs['y']['log_id'] = rep_i
            ### Standardized conditioning
            if args.p2p_impute:
                ### Inpaint with p2p
                traj_model_kwargs['y']['inpainted_motion'] = inpaint_traj_p2p.to(model_device)
                traj_model_kwargs['y']['inpainting_mask'] = inpaint_traj_mask_p2p.to(model_device)
            else:
                ### Inpaint with kps
                traj_model_kwargs['y']['inpainted_motion'] = inpaint_traj_points.to(model_device)
                traj_model_kwargs['y']['inpainting_mask'] = inpaint_traj_mask_points.to(model_device)

            # Set when to stop imputing
            traj_model_kwargs['y']['cond_until'] = impute_slack
            traj_model_kwargs['y']['impute_until'] = impute_until
            # NOTE: We have the option of switching the target motion from line to just key locations
            # We call this a 'second stage', which will start after t reach 'impute_until'
            traj_model_kwargs['y']['impute_until_second_stage'] = impute_slack
            traj_model_kwargs['y']['inpainted_motion_second_stage'] = inpaint_traj_points.to(model_device)
            traj_model_kwargs['y']['inpainting_mask_second_stage'] = inpaint_traj_mask_points.to(model_device)

            traj_diffusion.data_transform_fn = None
            traj_diffusion.data_inv_transform_fn = None
            traj_diffusion.log_trajectory_fn = partial(
                log_trajectory_from_xstart,
                kframes=kframes,
                inv_transform=data.dataset.t2m_dataset.inv_transform_th,
                abs_3d=args.abs_3d,
                traject_only=True,
                n_frames=cut_frames,
                combine_to_video=True,
                obs_list=obs_list)

            if args.guidance_mode == "kps":
                cond_fn_traj = CondKeyLocations(target=target,
                                            target_mask=target_mask,
                                            transform=data.dataset.t2m_dataset.transform_th,
                                            inv_transform=data.dataset.t2m_dataset.inv_transform_th,
                                            abs_3d=args.abs_3d,
                                            classifiler_scale=args.classifier_scale,
                                            use_mse_loss=args.gen_mse_loss,
                                            use_rand_projection=False,
                                            )
            elif args.guidance_mode == "sdf":
                cond_fn_traj = CondKeyLocationsWithSdf(target=target,
                                        target_mask=target_mask,
                                        transform=data.dataset.t2m_dataset.transform_th,
                                        inv_transform=data.dataset.t2m_dataset.inv_transform_th,
                                        abs_3d=args.abs_3d,
                                        classifiler_scale=args.classifier_scale,
                                        use_mse_loss=args.gen_mse_loss,
                                        use_rand_projection=False,
                                        obs_list=obs_list
                                        )
            else:
                cond_fn_traj = None

            sample_fn = traj_diffusion.p_sample_loop
            dump_steps = [1, 100, 300, 500, 700, 850, 999]
            traj_sample = sample_fn(
                traj_model,
                (args.batch_size, traj_model.njoints, traj_model.nfeats,
                 n_frames),
                clip_denoised=True,  # False,
                model_kwargs=traj_model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None, # None,
                progress=True,
                dump_steps=dump_steps,  # None,
                noise=None,
                const_noise=False,
                cond_fn=cond_fn_traj,
            )

            # Set inpainting information for motion model
            traj_motion, traj_mask = get_inpainting_motion_from_traj(
                traj_sample[-1],
                inv_transform_fn=data.dataset.t2m_dataset.inv_transform_th,
            )
            # plt.scatter(traj_motion[0, 1, 0, :120].cpu().numpy(), traj_motion[0, 2, 0, :120].cpu().numpy())
            model_kwargs['y']['inpainted_motion'] = traj_motion
            model_kwargs['y']['inpainting_mask'] = traj_mask
            # Assume the target has dimention [bs, 120, 22, 3] in case we do key poses instead of key location
            target = torch.zeros([args.batch_size, n_frames, 22, 3], device=traj_motion.device)
            target_mask = torch.zeros_like(target, dtype=torch.bool)
            # This assume that the traj_motion is in the 3D space without normalization
            # traj_motion: [3, 263, 1, 196]
            target[:, :, 0, [0, 2]] = traj_motion.permute(0, 3, 2, 1)[:, :, 0,[1, 2]]
            # target_mask[:, :int(flag.GEN_MOTION_LENGTH_CUT * 20.0), 0, [0, 2]] = True
            target_mask[:, :, 0, [0, 2]] = True

        elif not args.guidance_mode == "mdm_legacy":
            model_kwargs['y']['inpainted_motion'] = inpaint_motion_points.to(model_device) # inpaint_motion_p2p
            model_kwargs['y']['inpainting_mask'] = inpaint_mask_points.to(model_device)  # inpaint_p2p_mask

        if args.use_ddim:
            sample_fn = diffusion.ddim_sample_loop
            # dump_steps for logging progress
            dump_steps = [0, 1, 10, 30, 50, 70, 99]
        else:
            sample_fn = diffusion.p_sample_loop
            # dump_steps = [1, 100, 300, 500, 700, 850, 999]
            dump_steps = [999]

        # NOTE: Delete inpainting information if not using it. Just to be safe
        # TODO: remove this
        if not args.do_inpaint and "inpainted_motion" in model_kwargs['y'].keys():
            del model_kwargs['y']['inpainted_motion']
            del model_kwargs['y']['inpainting_mask']

        # Name for logging
        model_kwargs['y']['log_id'] = rep_i
        model_kwargs['y']['cond_until'] = motion_cond_until  # impute_slack
        model_kwargs['y']['impute_until'] = motion_impute_until # 20  # impute_slack
        # Pass functions to the diffusion
        diffusion.data_get_mean_fn = data.dataset.t2m_dataset.get_std_mean
        diffusion.data_transform_fn = data.dataset.t2m_dataset.transform_th
        diffusion.data_inv_transform_fn = data.dataset.t2m_dataset.inv_transform_th
        diffusion.log_trajectory_fn = partial(
            log_trajectory_from_xstart,
            kframes=kframes,
            inv_transform=data.dataset.t2m_dataset.inv_transform_th,
            abs_3d=args.abs_3d,
            use_rand_proj=args.use_random_proj,
            traject_only=args.traj_only,
            n_frames=cut_frames,
            combine_to_video=True,
            obs_list=obs_list)
        
        # diffusion.log_trajectory_fn(input_motions.detach(), out_path, [1000], torch.tensor([1000] * args.batch_size), model_kwargs['y']['log_id'])
        # diffusion.log_trajectory_fn(model_kwargs['y']['inpainted_motion'].detach(), out_path, [1000], torch.tensor([1000] * args.batch_size), model_kwargs['y']['log_id'])
        
        # TODO: move the followings to a separate function
        if args.guidance_mode == "kps" or args.guidance_mode == "trajectory":
            cond_fn = CondKeyLocations(target=target,
                                        target_mask=target_mask,
                                        transform=data.dataset.t2m_dataset.transform_th,
                                        inv_transform=data.dataset.t2m_dataset.inv_transform_th,
                                        abs_3d=args.abs_3d,
                                        classifiler_scale=args.classifier_scale,
                                        use_mse_loss=args.gen_mse_loss,
                                        use_rand_projection=args.use_random_proj
                                        )
        elif args.guidance_mode == "sdf":
            cond_fn = CondKeyLocationsWithSdf(target=target,
                                        target_mask=target_mask,
                                        transform=data.dataset.t2m_dataset.transform_th,
                                        inv_transform=data.dataset.t2m_dataset.inv_transform_th,
                                        abs_3d=args.abs_3d,
                                        classifiler_scale=args.classifier_scale,
                                        use_mse_loss=args.gen_mse_loss,
                                        use_rand_projection=args.use_random_proj,
                                        obs_list=obs_list
                                        )
        elif args.guidance_mode == "no" or args.guidance_mode == "mdm_legacy":
            cond_fn = None

        ###################
        # MODEL INFERENCING
        ###################
        # Set DDIM step to be the same as when editing
        # args.ddim_step = 25

        # list of [bs, njoints, nfeats, nframes] each element is a different time step
        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, n_frames),
            # clip_denoised=False,
            clip_denoised=not args.predict_xstart,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step # NOTE: testing this
            init_image=None, # input_motions,  # init_image, # None, # NOTE: testing this
            progress=True,
            dump_steps=dump_steps,  # None,
            noise=None,
            const_noise=False,
            cond_fn=cond_fn,
        )

        #######################
        ##### Edting here #####
        #######################

        # Visualize the generated motion
        gen_eff_len = min(sample[0].shape[-1], cut_frames)
        # Take last sample and cut to the target length
        gen_sample = sample[-1][:, :, :, :gen_eff_len]
        # Convert sample to motion
        gen_motions, cur_lengths, cur_texts = sample_to_motion(
            gen_sample, args, model_kwargs, model, gen_eff_len,
            data.dataset.t2m_dataset.inv_transform)
        gen_motion_vis = gen_motions[0][0].transpose(2, 0, 1) # [120, 22, 3]

        skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain
        animation_save_path = os.path.join(out_path, "before_edit.mp4")
        # task = "trajectory_editing"
        # task = "dense_optimization"
        task = "motion_projection"
        START_FROM_NOISE = False
        KEEP_NOISE_SAME = True

        if task == "trajectory_editing":
            # Get obstacle list
            obs_list = get_obstacles()
            # For trajectory editing, obtain the frame indices and the target locations
            selected_index, target_locations = plot_3d_motion_static(animation_save_path,
                            skeleton,
                            gen_motion_vis,
                            dataset=args.dataset,
                            title=cur_texts,
                            fps=fps,
                            traj_only=args.traj_only,
                            kframes=kframes,
                            obs_list=obs_list,
                            # NOTE: TEST
                            target_pose=input_skels[0].cpu().numpy(),
                            gt_frames=[kk for (kk, _) in kframes] if POSE_EDITING else []) # GUIDE_WITH_POSES
            # Set up the new target based on the selected frames and the target locations
            target = torch.zeros([args.batch_size, n_frames, 22, 3], device=model_device)
            target_mask = torch.zeros_like(target, dtype=torch.bool)
            kframes = [(tt, locs) for (tt, locs) in zip(selected_index, target_locations)]
            for (tt, locs) in zip(selected_index, target_locations):
                print("target at %d = %.1f, %.1f" % (tt, locs[0], locs[1]))
                target[0, tt, 0, [0, 2]] = torch.tensor([locs[0], locs[1]], dtype=torch.float32, device=target.device)
                target_mask[0, tt, 0, [0, 2]] = True

        elif task == "pose_editing":
            # Select a frame in gen_motion_vis then pull both hands up by about 50cm
            selected_index = 80
            # gen_motion_vis
        
        elif task == "dense_optimization":
            # Dense optimization
            START_FROM_NOISE = True
            KEEP_NOISE_SAME = False
            # Change target to the generated motion
            target[0, :gen_eff_len, :, :] = torch.from_numpy(gen_motion_vis).to(target.device)
            input_skels[0, :, :, :] = target[0, :gen_eff_len, :, :]
            target_mask[0, :gen_eff_len, :, :] = True
            # target_mask[0, :, :, :] = False
            # target_mask[0, 110, :, :] = True
            # kframes = [(110, (0.0, 0.0))]
            kframes = []
            # import pdb; pdb.set_trace()
        
        elif task == "motion_projection":
            # Dense optimization
            START_FROM_NOISE = True
            KEEP_NOISE_SAME = False
            target[0, :gen_eff_len, :, :] = torch.from_numpy(gen_motion_vis).to(target.device)
            # Add noise to target
            target = target + (torch.randn_like(target) - 0.5) * 0.01 # 0.03

            input_skels[0, :, :, :] = target[0, :gen_eff_len, :, :]
            target_mask[0, :gen_eff_len, :, :] = True
            kframes = []
        
        elif task == "motion_blending":
            import pdb; pdb.set_trace()
            # Inference again to get a second motion with different prompt
            model_kwargs['y']['text'] = ['a person is jumping to the right']
            # list of [bs, njoints, nfeats, nframes] each element is a different time step
            sample_2 = sample_fn(
                model,
                (args.batch_size, model.njoints, model.nfeats, n_frames),
                # clip_denoised=False,
                clip_denoised=not args.predict_xstart,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step # NOTE: testing this
                init_image=None, # input_motions,  # init_image, # None, # NOTE: testing this
                progress=True,
                dump_steps=dump_steps,  # None,
                noise=None,
                const_noise=False,
                cond_fn=cond_fn,
            )
            # Concat the two motions, the latter half from the first motion and first half from the second motion
            gen_eff_len = min(sample[0].shape[-1], cut_frames)
            gen_eff_len_2 = min(sample_2[0].shape[-1], cut_frames)
            gen_sample = torch.cat([sample[-1][:, :, :, :gen_eff_len//2], sample_2[-1][:, :, :, gen_eff_len_2//2:]], dim=-1)
            import pdb; pdb.set_trace()

        ####
        model_kwargs['y']['text'] = ['']
        inverse_step = 99 # 30
        cur_t = inverse_step  # 100 - inverse_step
        dump_steps = [0, 5, 10, 20, 30, 40, 49]
        
        # dump_steps = [0, 5, 10, 15, 20, 25, 29]
        inv_noise, pred_x0_list = diffusion.invert(model,
                                                   sample[-1].clone(), 
                                                   model_kwargs=model_kwargs, # model_kwargs['y']['text'],
                                                   dump_steps=dump_steps,
                                                   num_inference_steps=inverse_step,
                                                   )
        # Visualize the inversion process on the second row
        for ii in range(len(sample)):
            if sample[ii].shape[0] > 1:
                sample[ii][1] = pred_x0_list[ii][0]
            else:
                sample[ii] = torch.cat([sample[ii], pred_x0_list[ii]], dim=0)

        # import pdb; pdb.set_trace()
        # model_kwargs['y']['text'] = ['a person walks to the right']

        ### Optimize

        # opt_step = 1601 # 401
        # step_out_list = [0, 10, 50, 100, 200, 300, 400]
        # step_out_list = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95]
        # step_out_list = [int(aa * opt_step) for aa in step_out_list]
        inter_out = []
        # cur_t = [cur_t] * cur_xt.shape[0]
        
        # cur_t = torch.tensor([cur_t] * inv_noise.shape[0], device=inv_noise.device)
        
        ###### NOTE: test full unrolling
        opt_step = 1000 # 400 # 800 # 400 # 1000 # 500 # 100
        inter_out = []
        step_out_list = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.95]
        step_out_list = [int(aa * opt_step) for aa in step_out_list]
        step_out_list[-1] = opt_step - 1
        cond_fn_opt = CondKeyLocationsLoss(target=target,
                                    target_mask=target_mask,
                                    transform=data.dataset.t2m_dataset.transform_th,
                                    inv_transform=data.dataset.t2m_dataset.inv_transform_th,
                                    abs_3d=args.abs_3d,
                                    use_mse_loss=False, # args.gen_mse_loss,
                                    use_rand_projection=args.use_random_proj,
                                    obs_list=obs_list,
                                    )
        
        # Test starting from random noise
        # START_FROM_NOISE = True
        # KEEP_NOISE_SAME = False
        # import pdb; pdb.set_trace()
        
        cur_xt = inv_noise.detach().clone()
        if START_FROM_NOISE:
            # cur_xt = (torch.rand_like(cur_xt) - 0.5) * 0.1 # * 2.0 # 0.1
            cur_xt = torch.randn_like(cur_xt) * 0.1 # * 2.0 # 0.1
            # cur_xt = torch.rand_like(cur_xt) *  0.1

        # plt.hist(cur_xt.reshape(-1).detach().cpu().numpy(), bins=100)

        starting_xt = cur_xt.detach().clone()
        cur_xt = cur_xt.detach().requires_grad_()
        perturb_grad_scale = 1e-4 # 2e-4 # 1e-4
        loss_prog = []
        grad_norm_prog = []
        noise_diff_prog = []
        loss_noise_prog = []
        optimizer = torch.optim.Adam([cur_xt], lr=3e-2)
        # optimizer = torch.optim.Adam([cur_xt], lr=5e-2) # For generation from noise
        # optimizer = torch.optim.Adam([cur_xt], lr=1e-2)  # lr=5e-2
        # optimizer = torch.optim.SGD([cur_xt], lr=3e0, momentum=0.9)
        args.ddim_step = 50 # 25 # 10
        diffusion = create_gaussian_diffusion(args)
        lambda_init_noise = 0.0005 # 0.0005 # 0.01
        # model_kwargs['y']['text'] = ['a person is walking while clapping their hands']
        # model_kwargs['y']['text'] = ['a person is jumping']
        # model_kwargs['y']['scale'] = model_kwargs['y']['scale'] * 0.0 + 5.0
        pbar = tqdm(range(0, opt_step))
        for i in pbar: # tqdm(range(0, opt_step)):
            ### Do forward inference
            # [N, 263, 1, 196]
            final_out = diffusion.ddim_sample_loop_full_chain(
                model,
                (args.batch_size, model.njoints, model.nfeats, n_frames),
                clip_denoised=not args.predict_xstart,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=False, # True,
                dump_steps=dump_steps,  # None,
                noise=cur_xt, # NOTE: <-- the most important part
                cond_fn=cond_fn,
            )

            ### Compute loss on x_0
            loss_cond = cond_fn_opt(final_out, **model_kwargs)
            if KEEP_NOISE_SAME:
                # Compute noise diff loss
                noise_diff = (cur_xt - starting_xt).norm(p=2, dim=[1,2,3], keepdim=False)
                loss_noise = lambda_init_noise * noise_diff.mean()
            else:
                loss_noise = 0

            loss = loss_cond + loss_noise
            # loss_skating = """new contact reward, only penalize skating when in contact, does not require always in contact"""
            # marker_in_contact = Y_w[:, :, feet_marker_idx, 2].abs() < 0.05
            # contact_speed = (Y_l_speed[:, :, feet_marker_idx] - 0.075).clamp(min=0) * marker_in_contact[:, 1:-1, :]  # [b ,t ,f]
            # r_contact_feet_new = torch.exp(- contact_speed.mean(dim=[1, 2]))

            ### backpropagate to x_t
            # print('loss: %6f' % (loss.data))
            pbar.set_description("loss: %6f" % (loss.data))
            loss_prog.append(loss.data.detach().cpu().numpy())
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # save grad norm
            # import pdb; pdb.set_trace()
            grad_norm_prog.append(cur_xt.grad.norm(p=2, dim=[1,2,3], keepdim=False).detach().cpu().numpy())
            # import pdb; pdb.set_trace()

            # Perturbation is just random noise added
            perturbation = perturb_grad_scale * torch.randn_like(cur_xt) if perturb_grad_scale else 0
            cur_xt.data += perturbation
            # cur_xt.data = (1 - perturb_grad_scale) * perturb_grad_scale + perturbation

            # Compute difference between current and starting point
            noise_diff = (cur_xt - starting_xt).norm(p=2, dim=[1,2,3], keepdim=False)
            noise_diff_prog.append(noise_diff.detach().cpu().numpy())

            if i in step_out_list:
                print("save optimize at", i)
                inter_out.append(final_out.detach().clone())

        # Plot loss
        plt.figure()
        plt.plot(loss_prog)
        plt.legend(["loss"])
        plt.savefig(os.path.join(out_path, "loss_prog.png"))
        plt.close()
        # Plot grad norm
        plt.figure()
        plt.plot(grad_norm_prog)
        plt.legend(["grad_norm"])
        plt.savefig(os.path.join(out_path, "grad_norm_prog.png"))
        plt.close()
        # Plot noise diff
        plt.figure()
        plt.plot(noise_diff_prog)
        plt.legend(["noise_diff"])
        plt.savefig(os.path.join(out_path, "noise_diff_prog.png"))

        final_out = final_out.detach().clone()
        # Visualize the generated motion on the third row
        for ii in range(len(sample)):
            # if sample[ii].shape[0] > 1:
            #     sample[ii][1] = pred_x0_list[ii][0]
            # else:
            # sample[ii] = torch.cat([sample[ii], final_out], dim=0)
            sample[ii] = torch.cat([sample[ii], inter_out[ii]], dim=0)
        # NOTE: hack
        args.num_samples = 3

        ### test do optimization at these steps
        # other_t = [90, 80, 70, 60, 50, 40, 30, 20, 10]
        # opt_step = 400
        # for tt in other_t:
        #     inverse_step = tt
        #     cur_t = torch.tensor([tt] * inv_noise.shape[0], device=inv_noise.device)

        #     with torch.enable_grad():
        #         for i in tqdm(range(0, opt_step)):
                    
        #             # import pdb; pdb.set_trace()
        #             cur_xt = cur_xt.detach().requires_grad_()
        #             out_orig = diffusion.p_mean_variance(
        #                 model,
        #                 cur_xt,
        #                 cur_t,
        #                 model_kwargs=model_kwargs,
        #             )
        #             # print("done inference")
        #             # import pdb; pdb.set_trace()
        #             # pred_xstart = out_orig["pred_xstart"]
        #             # pred_x_start = model(inv_noise, diffusion._scale_timesteps(cur_t), **model_kwargs)

        #             # pred_x_start = pred_x_start.detach()
        #             # loss_sds = torch.norm(cur_xstart - pred_x_start)

        #             ## Compute objective loss
        #             # loss_obj = cond_fn_opt(inv_noise, **model_kwargs)
        #             # loss_obj = cond_fn_opt(pred_x_start, **model_kwargs)
                    
        #             # NOTE: Should this be normalized to be just a direction?
        #             gradient = cond_fn_opt(cur_xt, cur_t, out_orig, **model_kwargs)
        #             # import pdb; pdb.set_trace()
        #             # gradient = gradient / gradient.norm(p=2, dim=[1,2,3], keepdim=True)
        #             # input / input.norm(p, dim)
        #             cur_xt = cur_xt + 1.0 * gradient

        #             ## Optimize
        #             # loss = loss_obj + 0.5 * loss_sds
        #             # loss =  loss_sds
        #             # print('loss: %6f' % (loss.data))
        #             # optimizer.zero_grad()
        #             # loss.backward()
        #             # optimizer.step()

        #             # if i in step_out_list:
        #             #     print("save optimize at", i)
        #             #     inter_out.append(out_orig["pred_xstart"].detach().clone())
        #             #     # inter_out.append(cur_xt.detach().clone())

        #         if tt in [90, 80, 60, 50, 30, 20, 10]:
        #             print("save optimize at", i)
        #             inter_out.append(out_orig["pred_xstart"].detach().clone())
        #             # inter_out.append(cur_xt.detach().clone())
            
        #     # Forward 10 steps
        #     sample_new = sample_fn(
        #         model,
        #         (args.batch_size, model.njoints, model.nfeats, n_frames),
        #         # clip_denoised=False,
        #         clip_denoised=not args.predict_xstart,
        #         model_kwargs=model_kwargs,
        #         skip_timesteps=100 - inverse_step,  # 0 is the default value - i.e. don't skip any step # NOTE: testing this
        #         init_image=None, # input_motions,  # init_image, # None, # NOTE: testing this
        #         progress=True,
        #         dump_steps=[tt-10],  # None,
        #         noise=cur_xt,
        #         const_noise=False,
        #         cond_fn=cond_fn,
        #     )
            
        #     cur_xt = sample_new[0].detach().clone()
        ### End test do optimization at these steps


        # # Do final inference
        # sample_new = sample_fn(
        #     model,
        #     (args.batch_size, model.njoints, model.nfeats, n_frames),
        #     # clip_denoised=False,
        #     clip_denoised=not args.predict_xstart,
        #     model_kwargs=model_kwargs,
        #     skip_timesteps=100 - 10, # inverse_step,  # 0 is the default value - i.e. don't skip any step # NOTE: testing this
        #     init_image=None, # input_motions,  # init_image, # None, # NOTE: testing this
        #     progress=True,
        #     dump_steps=dump_steps,  # None,
        #     noise=cur_xt,
        #     const_noise=False,
        #     cond_fn=cond_fn,
        # )
        # inter_out[-1] = sample_new[-1].detach().clone()

        # for ii in range(len(step_out_list)):
        #     sample[ii][2] = inter_out[ii][0]


        # ## use inverse noise to do forward again
        # sample_new = sample_fn(
        #     model,
        #     (args.batch_size, model.njoints, model.nfeats, n_frames),
        #     # clip_denoised=False,
        #     clip_denoised=not args.predict_xstart,
        #     model_kwargs=model_kwargs,
        #     skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step # NOTE: testing this
        #     init_image=None, # input_motions,  # init_image, # None, # NOTE: testing this
        #     progress=True,
        #     dump_steps=dump_steps,  # None,
        #     noise=inv_noise,
        #     const_noise=False,
        #     cond_fn=cond_fn,
        # )
        # for ii in range(len(sample) - 1):
        #     sample[ii + 1][2] = sample_new[ii][0]


        # Cut the generation to the desired length
        # NOTE: this is important for UNETs where the input must be specific size (e.g. 224)
        # but the output can be cut to any length
        gen_eff_len = min(sample[0].shape[-1], cut_frames)
        print('cut the motion length to', gen_eff_len)
        for j in range(len(sample)):
            sample[j] = sample[j][:, :, :, :gen_eff_len]
        ###################

        num_dump_step = len(dump_steps)
        args.num_dump_step = num_dump_step
        # Convert sample to XYZ skeleton locations
        # Each return size [bs, 1, 3, 120]
        cur_motions, cur_lengths, cur_texts = sample_to_motion(
            sample, args, model_kwargs, model, gen_eff_len,
            data.dataset.t2m_dataset.inv_transform)
        if task == "motion_projection":
            # import pdb; pdb.set_trace()
            # Visualize noisy motion in the second row last column
            cur_motions[-1][1] = target[0, :gen_eff_len, :, :].detach().cpu().numpy().transpose(1, 2, 0)
        all_motions.extend(cur_motions)
        all_lengths.extend(cur_lengths)
        all_text.extend(cur_texts)

    ### Save videos
    total_num_samples = args.num_samples * args.num_repetitions * num_dump_step

    # After concat -> [r1_dstep_1, r2_dstep_1, r3_dstep_1, r1_dstep_2, r2_dstep_2, ....]
    all_motions = np.concatenate(all_motions,
                                 axis=0)  # [bs * num_dump_step, 1, 3, 120]
    all_motions = all_motions[:total_num_samples]  # #       not sure? [bs, njoints, 6, seqlen]
    all_text = all_text[:
                        total_num_samples]  # len() = args.num_samples * num_dump_step
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(
        npy_path, {
            'motion': all_motions,
            'text': all_text,
            'lengths': all_lengths,
            'num_samples': args.num_samples,
            'num_repetitions': args.num_repetitions
        })
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain
    if args.traj_only:
        skeleton = [[0, 0]]

    sample_files = []
    num_samples_in_out_file = num_dump_step  # 7

    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)

    # NOTE: we change the behavior of num_samples to support visualising denoising progress with multiple dump steps
    # for sample_i in range(args.num_samples * num_dump_step): # range(args.num_samples):
    # for sample_i in range(args.num_repetitions): # range(args.num_samples):
    for sample_i in range(args.num_samples):
        rep_files = []
        # for rep_i in range(args.num_repetitions):
        # for rep_i in range(num_dump_step):
        for dump_step_i in range(num_dump_step):
            # idx = rep_i + sample_i * num_dump_step # rep_i*args.batch_size + sample_i
            # idx = sample_i * num_dump_step + dump_step_i
            idx = sample_i + dump_step_i * args.num_samples
            print("saving", idx)
            caption = all_text[idx]
            length = all_lengths[idx]
            motion = all_motions[idx].transpose(2, 0,
                                                1)[:length]  # [120, 22, 3]
            save_file = sample_file_template.format(sample_i, dump_step_i)
            print(
                sample_print_template.format(caption, sample_i, dump_step_i,
                                             save_file))
            animation_save_path = os.path.join(out_path, save_file)
            # import pdb; pdb.set_trace()
            plot_3d_motion(animation_save_path,
                           skeleton,
                           motion,
                           dataset=args.dataset,
                           title=caption,
                           fps=fps,
                           traj_only=args.traj_only,
                           kframes=kframes,
                           obs_list=obs_list,
                           # NOTE: TEST
                           target_pose=input_skels[0].cpu().numpy(),
                           gt_frames=[kk for (kk, _) in kframes] if POSE_EDITING else []) # GUIDE_WITH_POSES
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
            rep_files.append(animation_save_path)

        sample_files = save_multiple_samples(
            args, out_path, row_print_template, all_print_template,
            row_file_template, all_file_template, caption,
            num_samples_in_out_file, rep_files, sample_files, sample_i)

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


def do_sds_loss():
    pass
    ### SDS
    # import pdb; pdb.set_trace()
    # opt_step = 400
    # cur_xstart = sample[0].clone() * 0.0
    # # cur_xstart = torch.rand_like(cur_xstart)
    # cur_xstart.requires_grad = True

    # ## Optimize for global orientation
    # optimizer = torch.optim.Adam([cur_xstart], lr=5e-2)  # lr=1e-2
    # # loss_fn = torch.nn.MSELoss()
    # # loss_l1 = torch.nn.L1Loss()
    # ## TODO: need to rewrite the cond_fn for optimization
    # cond_fn_opt = CondKeyLocationsLoss(target=target,
    #                                 target_mask=target_mask,
    #                                 transform=data.dataset.t2m_dataset.transform_th,
    #                                 inv_transform=data.dataset.t2m_dataset.inv_transform_th,
    #                                 abs_3d=args.abs_3d,
    #                                 use_mse_loss=args.gen_mse_loss,
    #                                 use_rand_projection=args.use_random_proj
    #                                 )
    # # import pdb; pdb.set_trace()
    # # loss = cond_fn_opt(cur_xstart, **model_kwargs)
    # # print('loss: %6f' % (loss.data))
    # sds_at_t = torch.tensor([200] * cur_xstart.shape[0], device=cur_xstart.device)
    
    # # noise = torch.randn_like(cur_xstart)

    # # import pdb; pdb.set_trace()
    # for i in range(0, opt_step):
    #     ## Optimize the motion with SDS and objective function
    #     # sds_at_t = torch.rand(cur_xstart.shape[0], device=cur_xstart.device) * 1000
    #     # sds_at_t = sds_at_t.long()
    #     # ## Do q_sample
    #     # # import pdb; pdb.set_trace()
    #     cur_xt = diffusion.q_sample(cur_xstart, sds_at_t, noise=None)
    #     pred_x_start = model(cur_xt, diffusion._scale_timesteps(sds_at_t), **model_kwargs)

    #     ## Compute SDS loss
    #     ## Stop grad from q_sample
    #     pred_x_start = pred_x_start.detach()
    #     loss_sds = torch.norm(cur_xstart - pred_x_start)

    #     ## Compute objective loss
    #     loss_obj = cond_fn_opt(cur_xstart, **model_kwargs)
    #     # loss_obj = cond_fn_opt(pred_x_start, **model_kwargs)

    #     ## Optimize
    #     # loss = loss_obj + 0.5 * loss_sds
    #     loss =  loss_sds
    #     print('loss: %6f' % (loss.data))
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    # # import pdb; pdb.set_trace()
    # sample.append(cur_xstart.detach())
    # dump_steps.append(-300)
    # # sample[0] = cur_xstart.detach()
    #######################


def load_dataset(args, max_frames, n_frames):
    conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=max_frames,
        split='test',
        hml_mode='text_only', # 'train'
        use_abs3d=args.abs_3d,
        traject_only=args.traj_only,
        use_random_projection=args.use_random_proj,
        random_projection_scale=args.random_proj_scale,
        augment_type='none',
        std_scale_shift=args.std_scale_shift,
        drop_redundant=args.drop_redundant,
    )
    data = get_dataset_loader(conf)
    # what's this for?
    data.fixed_length = n_frames
    return data


def load_processed_file(model_device, batch_size, traject_only=False):
    '''Load template file for trajectory imputing'''
    template_path = "./assets/template_joints.npy"
    init_joints = torch.from_numpy(np.load(template_path))
    from data_loaders.humanml.scripts.motion_process import process_file, recover_root_rot_pos
    data, ground_positions, positions, l_velocity = process_file(
        init_joints.permute(0, 3, 1, 2)[0], 0.002)
    init_image = data
    # make it (1, 263, 1, 120)
    init_image = torch.from_numpy(init_image).unsqueeze(0).float()
    init_image = torch.cat([init_image, init_image[0:1, 118:119, :].clone()],
                           dim=1)
    # Use transform_fn instead
    # init_image = (init_image - data.dataset.t2m_dataset.mean) / data.dataset.t2m_dataset.std
    init_image = init_image.unsqueeze(1).permute(0, 3, 1, 2)
    init_image = init_image.to(model_device)
    if traject_only:
        init_image = init_image[:, :4, :, :]

    init_image = init_image.repeat(batch_size, 1, 1, 1)
    return init_image, ground_positions


if __name__ == "__main__":
    main()
