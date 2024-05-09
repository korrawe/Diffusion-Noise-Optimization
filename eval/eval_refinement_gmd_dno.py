from utils.parser_util import eval_args, EvalArgs
from utils.fixseed import fixseed
from datetime import datetime
from data_loaders.humanml.motion_loaders.model_motion_loaders import get_mdm_loader  # get_motion_loader
from data_loaders.humanml.utils.metrics import *
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from collections import OrderedDict
from data_loaders.humanml.scripts.motion_process import *
from data_loaders.humanml.utils.utils import *
from utils.model_util import create_model_and_diffusion, create_gaussian_diffusion, load_model_wo_clip, load_saved_model
from diffusion import logger
from utils import dist_util
from data_loaders.get_data import DatasetConfig, get_dataset, get_dataset_loader
from model.cfg_sampler import ClassifierFreeSampleModel
from sample.noise_optimizer import NoiseOptimizer, NoiseOptOptions
import os.path as osp
# For testing
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.humanml.utils.metrics import calculate_skating_ratio
from data_loaders.humanml.data.dataset import sample_to_motion
import torch
from sample.condition import CondKeyLocationsLoss
from data_loaders.humanml.scripts.motion_process import process_file
from torch.utils.data import Subset, TensorDataset, DataLoader
import random
from eval.calculate_fid import calculate_fid_given_two_populations
from data_loaders.humanml.data.dataset import abs3d_to_rel

torch.multiprocessing.set_sharing_strategy('file_system')


def evaluate_matching_score(eval_wrapper, motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    skating_ratio_dict = OrderedDict({})
    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        skate_ratio_sum = 0.0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                if motion_loader_name == "vald":
                    word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _, skate_ratio = batch
                else:
                    word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens)
                dist_mat = euclidean_distance_matrix(
                    text_embeddings.cpu().numpy(),
                    motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

                if motion_loader_name == "vald":
                    skate_ratio_sum += skate_ratio.sum()

            all_motion_embeddings = np.concatenate(all_motion_embeddings,
                                                   axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings
        
        if motion_loader_name == "vald":
            # For skating evaluation
            skating_score = skate_ratio_sum / all_size
            skating_ratio_dict[motion_loader_name] = skating_score
            print(f'---> [{motion_loader_name}] Skating Ratio: {skating_score:.4f}')
            print(f'---> [{motion_loader_name}] Skating Ratio: {skating_score:.4f}', file=file, flush=True)

        print(
            f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}'
        )
        print(
            f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}',
            file=file,
            flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i + 1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict, skating_ratio_dict


def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            _, _, _, sent_lens, motions, m_lens, _ = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions, m_lens=m_lens)
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # print(mu)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file, diversity_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}',
              file=file,
              flush=True)
    return eval_dict


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, file,
                           mm_num_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions, m_lens, trajs = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(
                    motions[0], m_lens[0])
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings,
                                             dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings,
                                                    mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}',
              file=file,
              flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation_old(eval_wrapper,
               gt_loader,
               eval_motion_loaders,
               log_file,
               replication_times,
               diversity_times,
               mm_num_times,
               run_mm=False):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({
            'Matching Score': OrderedDict({}),
            'R_precision': OrderedDict({}),
            'FID': OrderedDict({}),
            'Diversity': OrderedDict({}),
            'MultiModality': OrderedDict({}),
            'Skating Ratio': OrderedDict({})
        })
        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items(
            ):
                # NOTE: set the seed for each motion loader based on the replication number
                motion_loader, mm_motion_loader = motion_loader_getter(
                    seed=replication)
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(
                f'==================== Replication {replication} ===================='
            )
            print(
                f'==================== Replication {replication} ====================',
                file=f,
                flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict, skating_ratio_dict = evaluate_matching_score(
                eval_wrapper, motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict,
                                          f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f, diversity_times)

            if run_mm:
                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                mm_score_dict = evaluate_multimodality(eval_wrapper,
                                                       mm_motion_loaders, f,
                                                       mm_num_times)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in skating_ratio_dict.items():
                if key not in all_metrics['Skating Ratio']:
                    all_metrics['Skating Ratio'][key] = [item]
                else:
                    all_metrics['Skating Ratio'][key] += [item]

            for key, item in mat_score_dict.items():
                if key not in all_metrics['Matching Score']:
                    all_metrics['Matching Score'][key] = [item]
                else:
                    all_metrics['Matching Score'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]
            if run_mm:
                for key, item in mm_score_dict.items():
                    if key not in all_metrics['MultiModality']:
                        all_metrics['MultiModality'][key] = [item]
                    else:
                        all_metrics['MultiModality'][key] += [item]

        # print(all_metrics['Diversity'])
        mean_dict = {}
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name,
                  file=f,
                  flush=True)
            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(
                    np.array(values), replication_times)
                mean_dict[metric_name + '_' + model_name] = mean
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(
                        mean, np.float32):
                    print(
                        f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}'
                    )
                    print(
                        f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}',
                        file=f,
                        flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (
                            i + 1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)
        return mean_dict


def compute_jitter(
    predicted_position,
    fps=20,
):
    cal_jitter = (
        (
            (
                predicted_position[3:]
                - 3 * predicted_position[2:-1]
                + 3 * predicted_position[1:-2]
                - predicted_position[:-3]
            )
            * (fps**3)
        )
        .norm(dim=2)
        .mean()
    )
    return cal_jitter


def load_dataset(args, n_frames, split, hml_mode, only_idx=None, with_loader=True, shuffle=True, drop_last=True):
    # (name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, hml_mode='gt'
    conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=n_frames,
        split=split,
        hml_mode=hml_mode,
        use_abs3d=args.abs_3d,
        traject_only=args.traj_only,
        use_random_projection=args.use_random_proj,
        random_projection_scale=args.random_proj_scale,
        augment_type='none',
        std_scale_shift=args.std_scale_shift,
        drop_redundant=args.drop_redundant,
    )
    data = get_dataset_loader(conf, only_idx=only_idx, with_loader=with_loader, shuffle=shuffle, drop_last=drop_last)
    if with_loader:
        # what's this for?
        data.fixed_length = n_frames
    return data


def input_to_motion(motion, gen_loader, model):
    gt_poses = motion.permute(0, 2, 3, 1)
    gt_poses = gt_poses * gen_loader.dataset.std + gen_loader.dataset.mean  # [bs, 1, 196, 263]
    gt_skels = recover_from_ric(gt_poses, 22, abs_3d=False).squeeze(1)
    gt_skels = gt_skels.permute(0, 2, 3, 1)
    gt_skels = model.rot2xyz(x=gt_skels, mask=None, pose_rep='xyz', glob=True, translation=True, 
                                jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None, get_rotations_back=False)
    gt_skels = gt_skels.permute(0, 3, 1, 2)
    return gt_skels


def plot_debug(motion_to_plot, name, gen_loader, length):
    plot_3d_motion(name, gen_loader.dataset.kinematic_chain, 
                   motion_to_plot[:length].detach().cpu().numpy(), 'length %d' % length, 'humanml', fps=20)



def calculate_results(gt_motions, target_motions, generated_motions, motion_lengths, target_joints):
    """
    Args:
        gt_motions: (tensor) [num_samples, 196, 22, 3]
        target_motions: (tensor) [num_samples, 196, 22, 3]
        generated_motions: (tensor) [num_samples, 196, 22, 3]
        motion_lengths: (list) [num_samples, ]
        target_joints: (list) [num_joints, ]
    """
    metrics = {
        "Foot skating": [],
        "Jitter": [],
        "MPJPE observed": [],
        "MPJPE all": [],
    }
    metrics_target = {
        "Foot skating": [],
        "Jitter": [],
        "MPJPE observed": [],
        "MPJPE all": [],
    }
    metrics_gt = {
        "Foot skating": [],
        "Jitter": [],
    }

    batch_size = len(gt_motions[0])
    # Compute norm of difference between gt_motions and target_motions

    for i in range(len(gt_motions)):
        # import pdb; pdb.set_trace()
        # MPJPE all
        # gt_target_diff = gt_motions[i] - target_motions[i]
        # gt_target_diff_norm_all = torch.norm(gt_target_diff, dim=3)

        # GT
        # import pdb; pdb.set_trace()
        gt_cut = gt_motions[i, :motion_lengths[i], :, :]
        skate_ratio, _ = calculate_skating_ratio(gt_cut.permute(1, 2, 0).unsqueeze(0)) # need input shape [bs, 22, 3, max_len]
        metrics_gt['Foot skating'].append(skate_ratio.item())
        metrics_gt['Jitter'].append(compute_jitter(gt_cut).item())

        # Target
        target_cut = target_motions[i, :motion_lengths[i], :, :]
        skate_ratio, _ = calculate_skating_ratio(target_cut.permute(1, 2, 0).unsqueeze(0))
        metrics_target['Foot skating'].append(skate_ratio.item())
        metrics_target['Jitter'].append(compute_jitter(target_cut).item())
        metrics_target['MPJPE all'].append(torch.norm(gt_cut - target_cut, dim=2).mean().item())
        metrics_target['MPJPE observed'].append(torch.norm(gt_cut[:, target_joints, :] - target_cut[:, target_joints, :], dim=2).mean().item())
        # metrics_target['MPJPE all'].append(gt_target_diff_norm_all[j, :motion_lengths[i][j], :].mean().item())
        # metrics_target['MPJPE observed'].append(gt_target_diff_norm_all[j, :motion_lengths[i][j], target_joints].mean().item())

        # Generated
        gen_cut = generated_motions[i, :motion_lengths[i], :, :]
        skate_ratio, _ = calculate_skating_ratio(gen_cut.permute(1, 2, 0).unsqueeze(0))
        metrics['Foot skating'].append(skate_ratio.item())
        metrics['Jitter'].append(compute_jitter(gen_cut).item())
        metrics['MPJPE all'].append(torch.norm(gt_cut - gen_cut, dim=2).mean().item())
        metrics['MPJPE observed'].append(torch.norm(gt_cut[:, target_joints, :] - gen_cut[:, target_joints, :], dim=2).mean().item())
        
        # Compute error for xyz locations
        # cur_motion = sample_to_motion(sample, self.dataset, motion_model)
        # kps_error = compute_kps_error(cur_motion, gt_skel_motions, sampled_keyframes)  # [batch_size, 5] in meter
        # skate_ratio, skate_vel = calculate_skating_ratio(cur_motion)  # [batch_size]
    

    # import pdb; pdb.set_trace()
    return metrics, metrics_target, metrics_gt


def evaluation(
        args: EvalArgs,
        log_file,
        num_samples_limit,
        save_dir,
        noise_opt_conf,
        debug,
        added_noise_level=0.01,
    ):
    
    DEBUG = debug
    ### Task settings ###
    NOISY_TARGET = args.eval_project_noisy_target
    print(f"NOISY_TARGET: {NOISY_TARGET}")
    joint_to_condition = args.eval_project_num_joints  # 'all' or 'upper' or 'lower' or 'three' or 'five'
    print(f"joint_to_condition: {joint_to_condition}")
    if joint_to_condition == 'all':
        target_joints = [*range(22)]
    elif joint_to_condition == 'lower': # pelvis and knees and ankles
        target_joints = [0, 4, 5, 7, 8]
    elif joint_to_condition == 'three': # head and hands
        target_joints = [15, 20, 21]
    elif joint_to_condition == 'pthree': # adding pelvis
        target_joints = [0, 15, 20, 21]
    elif joint_to_condition == 'five': # head and hands and feet
        target_joints = [10, 11, 15, 20, 21]
    elif joint_to_condition == 'pfive': # adding pelvis
        target_joints = [0, 10, 11, 15, 20, 21]
    elif joint_to_condition == 'eight': # pelvis, head, hands, feet, shoulders
        target_joints = [0, 10, 11, 15, 16, 17, 20, 21]
    elif joint_to_condition == 'ten': # pelvis, head, hands, feet, knees, shoulders,
        target_joints = [0, 4, 5, 10, 11, 15, 16, 17, 20, 21]
    else:
        raise NotImplementedError(f"joint_to_condition [{joint_to_condition}] not implemented")
    # Joint description at ./data_loaders/humanml_utils.py
    #####################
    n_frames = 196
    device = dist_util.dev()

    # loop over the files in save_dir to filter out the ones that are already generated
    print(f"making dir: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    
    # random the indexes to generate from the dataset
    # fixed the random indexes (cannot later be changed without invalidating the cache)
    np.random.seed(0)
    split = 'test'
    full_dataset = load_dataset(args, args.num_frames, split, hml_mode='eval', with_loader=False)
    dataset_size = len(full_dataset)
    print(f"dataset_size: {dataset_size}")
    idx_to_generate = np.random.choice(dataset_size, size=num_samples_limit, replace=False)
    print(f"id_to_generate: {idx_to_generate}")

    # finding the indexes that are not in the cache (that need to be generated)
    idx_not_in_cache = []

    def gen_motion_file(i):
        return f'motion_{i:05d}.pt'
    def target_motion_file(i):
        return f'target_{i:05d}.pt'
    def gt_motion_file(i):
        return f'gt_{i:05d}.pt'
    def hist_motion_file(i):
        return f'hist_{i:05d}.pt'

    for i in idx_to_generate:
        motion_path = os.path.join(save_dir, gen_motion_file(i))
        if not os.path.exists(motion_path):
            idx_not_in_cache.append(i)
        else:
            print(f"Motion [{motion_path}] already exists, skipping...")
    # create a dataloader with only the indexes that need to be generated
    idx_dataset = TensorDataset(torch.tensor(idx_not_in_cache))
    idx_loader = DataLoader(idx_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # create a dataset with only the indexes that need to be generated
    gen_loader = load_dataset(args, args.num_frames, split, hml_mode='eval', only_idx=idx_not_in_cache, shuffle=False, drop_last=False)
    assert len(idx_loader.dataset) == len(gen_loader.dataset), f"idx_loader.dataset {len(idx_loader.dataset)} != gen_loader.dataset {len(gen_loader.dataset)}"

    # loading the model
    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, gen_loader)
    diffusion_ori = diffusion
    args.ddim_step = noise_opt_conf.unroll_steps
    diffusion = create_gaussian_diffusion(args)

    print(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path, use_avg=args.eval_use_avg)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(
            model)  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    # Generate motions
    for (idx,) , (motion_rep, model_kwargs) in tqdm(zip(idx_loader, gen_loader)):
        # idx [bs, ]
        # motion [batch_size, 263, 1, 196]
        batch_size = len(motion_rep)
        gt_skels = input_to_motion(motion_rep, gen_loader, model)
        # plot_debug(gt_skels, osp.join(save_dir, f"gt_{motion_id}.mp4"), gen_loader, model_kwargs['y']['lengths'])

        def generate_det_noise(gen_shape, seeds):
            assert gen_shape[0] == len(seeds)
            noise = []
            for _seed in seeds:
                torch.manual_seed(_seed)
                noise.append(torch.randn(gen_shape[1:]))
            noise = torch.stack(noise, dim=0)
            return noise

        # Set target and mask
        target = gt_skels
        if NOISY_TARGET:
            # avoid correlation with the starting z
            seeds = idx + 1
            target = target + generate_det_noise(target.shape, seeds=seeds).to(target.device) * added_noise_level  # +- 1 cm
        # Set mask to match target lengths
        target_mask = torch.zeros_like(target, dtype=torch.bool)
        for j in range(batch_size):
            target_mask[j, :model_kwargs['y']['lengths'][j], target_joints, :] = True
        
        # save target to file
        for i in range(len(target)):
            torch.save(target[i].detach().cpu().clone(), os.path.join(save_dir, target_motion_file(idx[i])))

        target = target.to(device)
        target_mask = target_mask.to(device)

        # Optimize here 
        model_kwargs["y"]["text"] = [""]
        model_kwargs['y']['traj_model'] = False
        # add CFG scale to batch
        if args.guidance_param != 1.:
            model_kwargs['y']['scale'] = torch.ones(batch_size,
                                                    device=dist_util.dev()) * args.guidance_param

        opt_step = noise_opt_conf.opt_steps
        args.ddim_step = noise_opt_conf.unroll_steps
        args.use_ddim = True
        diffusion = create_gaussian_diffusion(args)

        START_FROM_NOISE = True
        if START_FROM_NOISE:
            use_deterministic_noise = True
            gen_shape = [batch_size, model.njoints, model.nfeats, n_frames]
            if use_deterministic_noise:
                seeds = idx
                # each noise is determined by the seed
                cur_xt = generate_det_noise(gen_shape, seeds=seeds).to(device)
            else:
                cur_xt = torch.randn(gen_shape).to(device)
            # cur_xt = (torch.rand_like(cur_xt) - 0.5) * 0.1  # * 2.0 # 0.1
            # cur_xt = torch.rand_like(cur_xt) *  0.1
        else:
            # TODO: Test getting initial joints from inversion
            # Get motion rep from (noisy) target joints
            data_list = []
            for jj in range(batch_size):
                # Duplicate last frame
                target_dup = torch.cat([target[jj], target[jj][-1:]], dim=0).detach().cpu()
                # target_dup = torch.cat([gt_skels[jj], gt_skels[jj][-1:]], dim=0).detach().cpu()
                data_rep, ground_positions, positions, l_velocity = process_file(target_dup, 0.002)
                # data_rep is correct but need to scale with the same mean and variance
                data_rep = (data_rep - gen_loader.dataset.mean) / gen_loader.dataset.std   # [bs, 1, 196, 263]
                data_list.append(torch.from_numpy(data_rep).permute(1, 0).unsqueeze(1))
            target_rep = torch.stack(data_list, dim=0)
            # Test converting motion rep back to skeleton
            # tar_from_joint = sample_to_motion(target_rep, gen_loader.dataset, model, abs_3d=args.abs_3d)
            # tar_from_joint = tar_from_joint.permute(0, 3, 1, 2)
            # plot_debug(tar_from_joint, osp.join(save_dir, f"{motion_id}_target.mp4"), gen_loader, motion_lengths[i], idx=0)
            # plot_debug(tar_from_joint, osp.join(save_dir, f"{motion_id}_rep.mp4"), gen_loader, motion_lengths[i], idx=0)
            # import pdb; pdb.set_trace()

            # Do inversion
            # args.ddim_step = noise_opt_conf.unroll_steps
            # args.use_ddim = True
            # # import pdb; pdb.set_trace()
            # diffusion = create_gaussian_diffusion(args)
            model_kwargs["y"]["text"] = [""]
            inv_noise = diffusion_ori.invert(
                model,
                target_rep.clone().float().to(device),
                model_kwargs=model_kwargs,  # model_kwargs['y']['text'],
                dump_steps=[], # dump_steps,
                num_inference_steps=19 # 49, # 99,
            )
            # import pdb; pdb.set_trace()

            cur_xt = inv_noise.detach().clone()
            # repeat 
            # cur_xt = cur_xt.repeat(num_trials, 1, 1, 1)

        # Copy from gen_edit2.py
        loss_fn = CondKeyLocationsLoss(
            target=target,
            target_mask=target_mask,
            motion_length=model_kwargs['y']['lengths'],
            transform=gen_loader.dataset.t2m_dataset.transform_th,
            inv_transform=gen_loader.dataset.t2m_dataset.inv_transform_th,
            abs_3d=args.abs_3d,
            use_mse_loss=False,  # args.gen_mse_loss,
            use_rand_projection=args.use_random_proj,
            obs_list=[], # No obsatcles
        )
        criterion = lambda x: loss_fn(x, **model_kwargs)
        solver = lambda z: diffusion.ddim_sample_loop_full_chain(
            model,
            (batch_size, model.njoints, model.nfeats, n_frames),
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=False,  # True,
            noise=z,  # NOTE: <-- the most important part
            cond_fn=None,
        )
        # start optimizing
        noise_opt = NoiseOptimizer(
            model=solver,
            criterion=criterion,
            start_z=cur_xt,
            conf=noise_opt_conf,
            # make the noise optimization deterministic
            noise_seeds=idx,
            # noise_seeds=None,
        )
        out = noise_opt()
        with torch.no_grad():
            final_out = solver(out['z'])

        # loop over the final_out and save to file one by one
        for i in range(len(final_out)):
            torch.save(final_out[i].detach().cpu().clone(), os.path.join(save_dir, gen_motion_file(idx[i])))
            # save hist
            hist = out['hist'][i]
            # to keep the file size small, we only use the stats
            for k in ['x', 'z']:
                del hist[k]
            torch.save(hist, os.path.join(save_dir, hist_motion_file(idx[i])))

    # load from files
    generated_motions, target_motions, gt_motions = [], [], []
    motion_lengths = []
    # load the generated motions from files
    idx_dataset = TensorDataset(torch.tensor(idx_to_generate))
    idx_loader = DataLoader(idx_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    gen_loader = load_dataset(args, args.num_frames, split, hml_mode='eval', only_idx=idx_to_generate, shuffle=False, drop_last=False)
    assert len(idx_loader.dataset) == len(gen_loader.dataset), f"idx_loader.dataset {len(idx_loader.dataset)} != gen_loader.dataset {len(gen_loader.dataset)}"
    for (idx,), (motion_rep, model_kwargs) in tqdm(zip(idx_loader, gen_loader)):

        # save the gt file (for reference)
        for i in range(len(motion_rep)):
            gt_file = os.path.join(save_dir, gt_motion_file(idx[i]))
            # if gt file doesn't exist create one and save
            if not os.path.exists(gt_file):
                torch.save(motion_rep[i].detach().cpu().clone(), gt_file)

        gt_skels = input_to_motion(motion_rep, gen_loader, model)
        # Get GT motion skeletons. gt_skels [batch size, 196, 22, 3]
        gt_motions.append(gt_skels)
        motion_lengths.append(model_kwargs['y']['lengths'])

        # load from files
        for i in idx:
            # load the target
            target = torch.load(os.path.join(save_dir, target_motion_file(i)))
            target_motions.append(target)
            # load the generated motion
            generated = torch.load(os.path.join(save_dir, gen_motion_file(i)))
            # convert the generated motion to skeleton
            generated = sample_to_motion(generated.unsqueeze(0), gen_loader.dataset, model, abs_3d=args.abs_3d)
            generated = generated.permute(0, 3, 1, 2)
            generated_motions.append(generated)

            if DEBUG:
                hist = torch.load(os.path.join(save_dir, hist_motion_file(i)))
                # plot the optimization stats
                for key in [
                    "loss",
                    "loss_diff",
                    "loss_decorrelate",
                    "grad_norm",
                    "grad_norm_after",
                    "lr",
                    "perturb_scale",
                    "diff_norm",
                ]:
                    motion_id = f'{i:05d}'
                    plt.figure()
                    if key == "loss":
                        plt.semilogy(hist["step"], hist[key])
                        plt.ylim(top=0.4)
                        # Plot horizontal red line at lowest point of loss function
                        min_loss = min(hist["loss"])
                        plt.axhline(y=min_loss, color='r')
                        plt.text(0, min_loss, f"Min Loss: {min_loss:.4f}", color='r')
                    else:
                        plt.plot(hist["step"], hist[key])
                    plt.title(f"Motion #{motion_id}: {key}")
                    plt.legend([key])
                    plt.savefig(os.path.join(save_dir, f"hist_{motion_id}_{key}.png"))
                    plt.close()

    # (num_samples, x, x, x)
    generated_motions = torch.cat(generated_motions, dim=0)
    target_motions = torch.stack(target_motions, dim=0)
    gt_motions = torch.cat(gt_motions, dim=0)
    # (num_samples, )
    motion_lengths = torch.cat(motion_lengths, dim=0)
    
    # import pdb; pdb.set_trace()
    if DEBUG:
        for ii in range(len(generated_motions)):
            motion_id = f'{idx[ii]:05d}'
            plot_debug(generated_motions[ii], osp.join(save_dir, f"{motion_id}_gen.mp4"), gen_loader, motion_lengths[ii])
            plot_debug(target_motions[ii], osp.join(save_dir, f"{motion_id}_target.mp4"), gen_loader, motion_lengths[ii])
            plot_debug(gt_motions[ii], osp.join(save_dir, f"{motion_id}_gt.mp4"), gen_loader, motion_lengths[ii])
    # import pdb; pdb.set_trace()

    def calculate_fid():
        def load_all_motion_from_filenames(filenames):
            motion = []
            for filename in tqdm(filenames):
                motion.append(torch.load(filename))
            # [batch_size, 263, 1, 196]
            motion = torch.stack(motion, dim=0)
            return motion

        # list filenames
        gt_filenames = [os.path.join(save_dir, gt_motion_file(i)) for i in idx_to_generate]
        target_filenames = [os.path.join(save_dir, target_motion_file(i)) for i in idx_to_generate]
        gen_filenames = [os.path.join(save_dir, gen_motion_file(i)) for i in idx_to_generate]
        assert len(gt_filenames) == len(target_filenames) == len(gen_filenames), (
            f"len(gt_filenames) {len(gt_filenames)} != len(target_filenames)"
            f"{len(target_filenames)} != len(gen_filenames) {len(gen_filenames)}"
        )

        def convert_skel_to_motion_rep(skel, dataset):
            # skel [batch_size, 196, 22, 3]
            data_list = []
            for jj in range(len(skel)):
                # Duplicate last frame
                target_dup = torch.cat([skel[jj], skel[jj][-1:]], dim=0).detach().cpu()
                # target_dup = torch.cat([gt_skels[jj], gt_skels[jj][-1:]], dim=0).detach().cpu()
                data_rep, ground_positions, positions, l_velocity = process_file(target_dup, 0.002)
                # data_rep is correct but need to scale with the same mean and variance
                data_rep = (data_rep - dataset.mean) / dataset.std   # [bs, 1, 196, 263]
                data_list.append(torch.from_numpy(data_rep).permute(1, 0).unsqueeze(1))
            target_rep = torch.stack(data_list, dim=0)
            # [batch_size, 263, 1, 196]
            return target_rep

        # gt_motion = load_all_motion_from_filenames(gt_filenames)
        # # [batch_size, 196, 22, 3]
        # target_motion = load_all_motion_from_filenames(target_filenames)
        # target_motion = convert_skel_to_motion_rep(target_motion, full_dataset)
        # gen_motion = load_all_motion_from_filenames(gen_filenames)

        gt_motion = load_all_motion_from_filenames(gt_filenames)
        # gt_motion = convert_skel_to_motion_rep(gt_motions, full_dataset)
        
        # [batch_size, 196, 22, 3]
        target_motion = load_all_motion_from_filenames(target_filenames)
        target_motion = convert_skel_to_motion_rep(target_motion, full_dataset)
        # absolute
        # [bs, 263, 1, 196]
        gen_motion = load_all_motion_from_filenames(gen_filenames)
        # remove proj and abs
        gen_motion = abs3d_to_rel(gen_motion, model_dataset, model)
        # gen_motion = convert_skel_to_motion_rep(generated_motions, full_dataset)

        def get_motion_length():
            idx_dataset = TensorDataset(torch.tensor(idx_to_generate))
            idx_loader = DataLoader(idx_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            gen_loader = load_dataset(args, args.num_frames, split, hml_mode='eval', only_idx=idx_to_generate, shuffle=False, drop_last=False)
            assert len(idx_loader.dataset) == len(gen_loader.dataset), f"idx_loader.dataset {len(idx_loader.dataset)} != gen_loader.dataset {len(gen_loader.dataset)}"
            motion_length = []
            for (idx,), (motion_rep, model_kwargs) in tqdm(zip(idx_loader, gen_loader)):
                motion_length.append(model_kwargs['y']['lengths'])
            motion_length = torch.cat(motion_length, dim=0)
            return motion_length

        def load_differnt_to_motions(idx_to_generate):
            all_idxs = [*range(dataset_size)] 
            idxs_left = list(set(all_idxs) - set(idx_to_generate))
            other_idxs = np.random.choice(idxs_left, size=len(idx_to_generate), replace=False)

            gen_loader = load_dataset(args, args.num_frames, split, hml_mode='eval', only_idx=other_idxs, shuffle=False, drop_last=False)
            motion_length = []
            motion = []
            for (motion_rep, model_kwargs) in tqdm(gen_loader):
                motion_length.append(model_kwargs['y']['lengths'])
                motion.append(motion_rep)
            motion_length = torch.cat(motion_length, dim=0)
            motion = torch.cat(motion, dim=0)
            return motion, motion_length
        
        gt_length = get_motion_length()
        gt2_motion, gt2_length = load_differnt_to_motions(idx_to_generate)

        fid_gt_gt2 = calculate_fid_given_two_populations(gt_motion, gt2_motion, gt_length, gt2_length, dataset=full_dataset, 
                                            dataset_name='humanml', device=device, batch_size=64)
        fid_gt_target = calculate_fid_given_two_populations(gt_motion, target_motion, gt_length, gt_length, dataset=full_dataset, 
                                            dataset_name='humanml', device=device, batch_size=64)
        fid_gt_gen = calculate_fid_given_two_populations(gt_motion, gen_motion, gt_length, gt_length, dataset=full_dataset, 
                                            dataset_name='humanml', device=device, batch_size=64)
        return {
            "fid_gt_gt2": fid_gt_gt2,
            "fid_gt_target": fid_gt_target,
            "fid_gt_gen": fid_gt_gen,
        }

    fids = calculate_fid()
    metrics, metrics_target, metrics_gt = calculate_results(gt_motions, target_motions, generated_motions, motion_lengths, target_joints)    
    with open(log_file, 'w') as f:
        for (name, eval_results) in zip(["Ground truth", "Target", "Prediction"], [metrics_gt, metrics_target, metrics]):
            print(f"==================== {name} ====================")
            print(f"==================== {name} ====================", file=f, flush=True)
            for metric_name, metric_values in eval_results.items():
                metric_values = np.array(metric_values)
                unit_name = ""
                if metric_name == "Jitter":
                    unit_name = "(m/s^3)"
                elif metric_name == "MPJPE observed" or metric_name == "MPJPE all":
                    unit_name = "(m)"
                elif metric_name == "Foot skating":
                    unit_name = "(ratio)"
                print(f"Metric [{metric_name} {unit_name}]: Mean {metric_values.mean():.4f}, Std {metric_values.std():.4f}")
                print(f"Metric [{metric_name} {unit_name}]: Mean {metric_values.mean():.4f}, Std {metric_values.std():.4f}", file=f, flush=True)
        for fid_name, fid_value in fids.items():
            print(f"FID [{fid_name}]: {fid_value:.4f}")
            print(f"FID [{fid_name}]: {fid_value:.4f}", file=f, flush=True)
    # import pdb; pdb.set_trace()

def run(noise_opt_conf, num_samples_limit=10, batch_size=10, joints='all', noisy=True, debug=False, added_noise_level=0.01):
    noise_opt_conf.postfix += f"_bs{batch_size}"
    #### Noise Optimization Config ####
    # Select task to evaluate
    task = "motion_projection"
    ##################################

    # args = evaluation_parser()
    args = eval_args()
    args.batch_size = batch_size
    args.num_frames = 196 # This must be 196!

    args.eval_project_num_joints = joints
    args.eval_project_noisy_target = noisy

    if args.eval_project_num_joints != 'all':
        task += f"_{args.eval_project_num_joints}"

    if not args.eval_project_noisy_target:
        task += "_clean"
    else:
        print(f"Added noise level: {added_noise_level}")
        if added_noise_level != 0.01:
            task += f"_noise{added_noise_level}"

    # replication_times = 3  # about ... Hrs
    # Not used in this experiment
    replication_times = None

    fixseed(args.seed)

    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model',
                                                      '').replace('.pt', '')
    # save_dir = os.path.join(os.path.dirname(args.model_path), f'eval_{task}_{niter}_opt{opt_steps}_unroll{noise_opt_conf.unroll_steps}')

    if args.eval_project_num_joints == 'all':
        # for compatibility reasons
        save_dir = os.path.join(os.path.dirname(args.model_path), f'eval_{task}_{noise_opt_conf.name}')
    else:
        save_dir = os.path.join(os.path.dirname(args.model_path), f'eval_{task}', noise_opt_conf.name)
    print(f"save_dir: {save_dir}")

    log_file = os.path.join(save_dir, f'eval_N{num_samples_limit}.txt')
    print('> Saving the generated motion to {}'.format(save_dir))

    # if args.guidance_param != 1.:
    #     log_file += f'_gscale{args.guidance_param}'
    print(f'> Will save to log file [{log_file}]')

    print(f'> Eval task [{task}]')
    # if args.eval_mode == 'debug':

    dist_util.setup_dist(args.device)
    logger.configure()

    logger.log("creating data loader...")


    evaluation(
        args=args, 
        log_file=log_file,
        num_samples_limit=num_samples_limit,
        save_dir=save_dir,
        noise_opt_conf=noise_opt_conf,
        debug=debug,
        added_noise_level=added_noise_level,
    )

    #### End ####
    # eval_motion_loaders = {
    #     ################
    #     ## HumanML3D Dataset##
    #     ################
    #     'vald':
    #     lambda seed: get_mdm_loader(
    #         model,
    #         diffusion,
    #         args.batch_size,
    #         gen_loader,
    #         mm_num_samples,
    #         mm_num_repeats,
    #         gt_loader.dataset.opt.max_motion_length,
    #         num_samples_limit,
    #         args.guidance_param,
    #         seed=seed,
    #         save_dir=save_dir,
    #         full_inpaint=args.full_traj_inpaint,
    #     )
    # }

    # eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
    # evaluation(eval_wrapper,
    #            gt_loader,
    #            eval_motion_loaders,
    #            log_file,
    #            replication_times,
    #            diversity_times,
    #            mm_num_times,
    #            run_mm=run_mm)

# set torch deterministic

if __name__ == '__main__':
    # speed up eval
    torch.set_num_threads(1)

    lr = 5e-2 # 5e-2 (adam), 0.02 (sgd)

    # no decorr, compare sign, unit, normal
    for num_limits in [300]:
        for grad_mode, decorr, std_af, warm_up, scheduler, steps, perturb, sep_ode, unroll in [
            # # the best setting
            ("unit", 1e3, False, 50, 'cosine', 500, 0, False, 10),
            # # abaltions
            # ("normal", 0, False, 50, 'cosine', 500, 0, True, 10),
            # ("normal", 1e3, False, 50, 'cosine', 500, 0, False, 10),
            # ("unit", 0, False, 50, 'cosine', 500, 0, False, 10),
            # ("unit", 1e3, False, 0, 'constant', 500, 0, False, 10),
            # ("unit", 1e3, False, 50, 'cosine', 300, 0, False, 10),
            # ("unit", 1e3, False, 50, 'cosine', 700, 0, False, 10),
            # ("unit", 1e3, False, 50, 'cosine', 500, 1e-4, False, 10),
            # ("unit", 1e3, False, 50, 'cosine', 500, 2e-4, False, 10),
            # ("unit", 1e3, False, 50, 'cosine', 500, 5e-4, False, 10),
            # ("unit", 1e3, False, 50, 'cosine', 500, 0, False, 5),
            # ("unit", 1e3, False, 50, 'cosine', 500, 0, False, 20),
        ]:
            for task, noisy, added_noise in [
                # # for ablation
                # ('all', True, 0.01),
                # # high noise
                # ('all', True, 0.05),
                # ('three', True, 0.05),
                # ('five', True, 0.05),
                # ('lower', True, 0.05),
                # new joints
                # ('pfive', True, 0.05),
                # ('eight', True, 0.05),
                # ('ten', True, 0.05),
                # ('pfive', False, 0.05),
                # ('eight', False, 0.05),
                ('ten', False, 0.05),
            ]:
                noise_opt_conf = NoiseOptOptions(
                    unroll_steps=unroll, # 10,  #
                    opt_steps=steps,  #
                    optimizer="adam", # adam
                    grad_mode=grad_mode, # sign, unit
                    lr=lr,
                    perturb_scale=perturb, 
                    diff_penalty_scale=0,
                    decorrelate_scale=decorr, # 2e6, 1e6 (for split grad), 100 (for combined grad),  1e3 (for unit grad, split)
                    standardize_z_after_step=std_af,
                    separate_backward_for_ode=sep_ode,
                    lr_warm_up_steps=warm_up,
                    lr_scheduler=scheduler,
                    lr_decay_steps=steps,
                    postfix="newx",
                )
                run(noise_opt_conf, num_samples_limit=num_limits, batch_size=60, 
                    joints=task, noisy=noisy, added_noise_level=added_noise, debug=False)
