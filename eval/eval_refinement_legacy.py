import os.path as osp
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, TensorDataset

from data_loaders.get_data import DatasetConfig, get_dataset_loader
from data_loaders.humanml.data.dataset import sample_to_motion
from data_loaders.humanml.scripts.motion_process import *
from data_loaders.humanml.scripts.motion_process import process_file

# get_motion_loader
from data_loaders.humanml.utils.metrics import *
from data_loaders.humanml.utils.metrics import calculate_skating_ratio

# For testing
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.humanml.utils.utils import *
from diffusion import logger
from eval.calculate_fid import calculate_fid_given_two_populations
from model.cfg_sampler import ClassifierFreeSampleModel
from sample.condition import CondKeyLocationsLoss
from sample.noise_optimizer import NoiseOptimizer, NoiseOptOptions
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import (
    create_gaussian_diffusion,
    create_model_and_diffusion,
    load_model_wo_clip
)
from utils.parser_util_legacy import eval_args
from eval.eval_shared import *

def ddim_loop_with_gradient(
    diffusion,
    model,
    shape,
    noise=None,
    clip_denoised=True,
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

def plot_debug(motion_to_plot, name, gen_loader, length, clean_root=None):
    if clean_root is not None:
        clean_root = clean_root[:length]
    plot_3d_motion(
        name,
        gen_loader.dataset.kinematic_chain,
        motion_to_plot[:length].detach().cpu().numpy(),
        "" % length,
        "humanml",
        fps=20,
        clean_root=clean_root,
    )
    # motion_to_plot[:length].detach().cpu().numpy(), 'length %d' % length, 'humanml', fps=20)


def evaluation(
    args,
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
    joint_to_condition = (
        args.eval_project_num_joints
    )  # 'all' or 'upper' or 'lower' or 'three' or 'five'
    print(f"joint_to_condition: {joint_to_condition}")
    target_joints = get_target_joints(joint_to_condition)
    # Joint description at ./data_loaders/humanml_utils.py
    #####################
    n_frames = 196
    device = dist_util.dev()

    # loop over the files in save_dir to filter out the ones that are already generated
    os.makedirs(save_dir, exist_ok=True)

    # random the indexes to generate from the dataset
    # fixed the random indexes (cannot later be changed without invalidating the cache)
    np.random.seed(0)
    split = "test"
    full_dataset = load_dataset(
        args, split, hml_mode="eval", with_loader=False
    )
    dataset_size = len(full_dataset)
    print(f"dataset_size: {dataset_size}")
    idx_to_generate = np.random.choice(
        dataset_size, size=num_samples_limit, replace=False
    )
    print(f"id_to_generate: {idx_to_generate}")

    # finding the indexes that are not in the cache (that need to be generated)
    idx_not_in_cache = []

    def gen_motion_file(i):
        return f"motion_{i:05d}.pt"

    def target_motion_file(i):
        return f"target_{i:05d}.pt"

    def gt_motion_file(i):
        return f"gt_{i:05d}.pt"

    def hist_motion_file(i):
        return f"hist_{i:05d}.pt"

    for i in idx_to_generate:
        motion_path = os.path.join(save_dir, gen_motion_file(i))
        if not os.path.exists(motion_path):
            idx_not_in_cache.append(i)
        else:
            idx_not_in_cache.append(i)
            print(f"Motion [{motion_path}] already exists, skipping...")
    # create a dataloader with only the indexes that need to be generated
    idx_dataset = TensorDataset(torch.tensor(idx_not_in_cache))
    idx_loader = DataLoader(
        idx_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # create a dataset with only the indexes that need to be generated
    gen_loader = load_dataset(
        args,
        split,
        hml_mode="eval",
        only_idx=idx_not_in_cache,
        shuffle=False,
        drop_last=False,
    )
    assert (
        len(idx_loader.dataset) == len(gen_loader.dataset)
    ), f"idx_loader.dataset {len(idx_loader.dataset)} != gen_loader.dataset {len(gen_loader.dataset)}"

    # loading the model
    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, gen_loader)
    diffusion_ori = diffusion
    args.ddim_step = noise_opt_conf.unroll_steps
    diffusion = create_gaussian_diffusion(args)

    print(f"Loading checkpoints from [{args.model_path}]...")
    # load_saved_model(model, args.model_path, use_avg=args.eval_use_avg)
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(
            model
        )  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    # Generate motions
    for (idx,), (motion_rep, model_kwargs) in tqdm(zip(idx_loader, gen_loader)):
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
        import pdb

        if NOISY_TARGET:
            # avoid correlation with the starting z
            seeds = idx + 1
            import pdb

            aa = (
                generate_det_noise(target.shape, seeds=seeds).to(target.device)
                * added_noise_level
            )
            target = (
                target
                + generate_det_noise(target.shape, seeds=seeds).to(target.device)
                * added_noise_level
            )  # +- 1 cm
        # Set mask to match target lengths
        target_mask = torch.zeros_like(target, dtype=torch.bool)
        for j in range(batch_size):
            target_mask[j, : model_kwargs["y"]["lengths"][j], target_joints, :] = True

        # save target to file
        for i in range(len(target)):
            torch.save(
                target[i].detach().cpu().clone(),
                os.path.join(save_dir, target_motion_file(idx[i])),
            )

        target = target.to(device)
        target_mask = target_mask.to(device)

        # Optimize here
        model_kwargs["y"]["text"] = [""]
        model_kwargs["y"]["traj_model"] = False
        # add CFG scale to batch
        if args.guidance_param != 1.0:
            model_kwargs["y"]["scale"] = (
                torch.ones(batch_size, device=dist_util.dev()) * args.guidance_param
            )

        opt_step = noise_opt_conf.opt_steps
        args.ddim_step = noise_opt_conf.unroll_steps
        args.use_ddim = True
        diffusion = create_gaussian_diffusion(args, timestep_respacing=f"ddim{args.ddim_step}")

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
                target_dup = (
                    torch.cat([target[jj], target[jj][-1:]], dim=0).detach().cpu()
                )
                # target_dup = torch.cat([gt_skels[jj], gt_skels[jj][-1:]], dim=0).detach().cpu()
                data_rep, ground_positions, positions, l_velocity = process_file(
                    target_dup, 0.002
                )
                # data_rep is correct but need to scale with the same mean and variance
                data_rep = (
                    data_rep - gen_loader.dataset.mean
                ) / gen_loader.dataset.std  # [bs, 1, 196, 263]
                data_list.append(torch.from_numpy(data_rep).permute(1, 0).unsqueeze(1))
            target_rep = torch.stack(data_list, dim=0)
            # Test converting motion rep back to skeleton
            # tar_from_joint = sample_to_motion(target_rep, gen_loader.dataset, model, abs_3d=args.abs_3d)
            # tar_from_joint = tar_from_joint.permute(0, 3, 1, 2)
            # plot_debug(tar_from_joint, osp.join(save_dir, f"{motion_id}_target.mp4"), gen_loader, motion_lengths[i], idx=0)
            # plot_debug(tar_from_joint, osp.join(save_dir, f"{motion_id}_rep.mp4"), gen_loader, motion_lengths[i], idx=0)
            # import pdb; pdb.()

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
                dump_steps=[],  # dump_steps,
                num_inference_steps=19,  # 49, # 99,
            )
            # import pdb; pdb.set_trace()

            cur_xt = inv_noise.detach().clone()
            # repeat
            # cur_xt = cur_xt.repeat(num_trials, 1, 1, 1)

        # Copy from gen_edit2.py
        loss_fn = CondKeyLocationsLoss(
            target=target,
            target_mask=target_mask,
            motion_length=model_kwargs["y"]["lengths"],
            transform=gen_loader.dataset.t2m_dataset.transform_th,
            inv_transform=gen_loader.dataset.t2m_dataset.inv_transform_th,
            abs_3d=False,
            use_mse_loss=False,  # args.gen_mse_loss,
            use_rand_projection=False,
            obs_list=[],  # No obsatcles
        )
        criterion = lambda x: loss_fn(x, **model_kwargs)
        # solver = lambda z: diffusion.ddim_sample_loop_full_chain(
        #     model,
        #     (batch_size, model.njoints, model.nfeats, n_frames),
        #     model_kwargs=model_kwargs,
        #     skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        #     init_image=None,
        #     progress=False,  # True,
        #     noise=z,  # NOTE: <-- the most important part
        #     cond_fn=None,
        # )
        def solver(z):
            return ddim_loop_with_gradient(
                diffusion,
                model,
                (batch_size, model.njoints, model.nfeats, n_frames),
                model_kwargs=model_kwargs,
                noise=z,
                gradient_checkpoint=False,
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
            final_out = solver(out["z"])

        # loop over the final_out and save to file one by one
        for i in range(len(final_out)):
            torch.save(
                final_out[i].detach().cpu().clone(),
                os.path.join(save_dir, gen_motion_file(idx[i])),
            )
            # save hist
            hist = out["hist"][i]
            # to keep the file size small, we only use the stats
            for k in ["x", "z"]:
                del hist[k]
            torch.save(hist, os.path.join(save_dir, hist_motion_file(idx[i])))

    # load from files
    generated_motions, target_motions, gt_motions = [], [], []
    motion_lengths = []
    # load the generated motions from files
    idx_dataset = TensorDataset(torch.tensor(idx_to_generate))
    idx_loader = DataLoader(
        idx_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    gen_loader = load_dataset(
        args,
        split,
        hml_mode="eval",
        only_idx=idx_to_generate,
        shuffle=False,
        drop_last=False,
    )
    assert (
        len(idx_loader.dataset) == len(gen_loader.dataset)
    ), f"idx_loader.dataset {len(idx_loader.dataset)} != gen_loader.dataset {len(gen_loader.dataset)}"
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
        motion_lengths.append(model_kwargs["y"]["lengths"])

        # load from files
        for i in idx:
            # load the target
            target = torch.load(os.path.join(save_dir, target_motion_file(i)))
            target_motions.append(target)
            # load the generated motion
            generated = torch.load(os.path.join(save_dir, gen_motion_file(i)))
            # convert the generated motion to skeleton
            generated = sample_to_motion(
                generated.unsqueeze(0), gen_loader.dataset, model, abs_3d=False
            )
            generated = generated.permute(0, 3, 1, 2)
            generated_motions.append(generated)

            if DEBUG and False:
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
                    motion_id = f"{i:05d}"
                    plt.figure()
                    if key == "loss":
                        plt.semilogy(hist["step"], hist[key])
                        plt.ylim(top=0.4)
                        # Plot horizontal red line at lowest point of loss function
                        min_loss = min(hist["loss"])
                        plt.axhline(y=min_loss, color="r")
                        plt.text(0, min_loss, f"Min Loss: {min_loss:.4f}", color="r")
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
            # if ii >= len(idx):
            #     break
            # motion_id = f'{idx[ii]:05d}'
            if ii > 20:
                break
            motion_id = f"{ii:05d}" + "i"
            plot_debug(
                generated_motions[ii],
                osp.join(save_dir, f"{motion_id}_gen.mp4"),
                gen_loader,
                motion_lengths[ii],
            )
            # import pdb; pdb.set_trace()
            # Hack to make the noisy target look smooth
            target_clone = target_motions[ii].clone()
            # target_clone[:, 0, :] = gt_motions[ii, :, 0, :]
            clean_root = gt_motions[ii, :, 0, :].detach().cpu().numpy()
            # import pdb; pdb.set_trace()
            # plot_debug(target_motions[ii], osp.join(save_dir, f"{motion_id}_target.mp4"), gen_loader, motion_lengths[ii])
            plot_debug(
                target_clone,
                osp.join(save_dir, f"{motion_id}_target.mp4"),
                gen_loader,
                motion_lengths[ii],
                clean_root=clean_root,
            )
            plot_debug(
                gt_motions[ii],
                osp.join(save_dir, f"{motion_id}_gt.mp4"),
                gen_loader,
                motion_lengths[ii],
            )
    import pdb

    SAVE_FOR_VIS = True
    if SAVE_FOR_VIS:
        # Refined motion
        import pdb

        npy_path = os.path.join(save_dir, "results.npy")
        all_motions = generated_motions.permute(0, 2, 3, 1).detach().cpu().numpy()
        print(f"saving results file to [{npy_path}]")
        np.save(
            npy_path,
            {
                "motion": all_motions,
                "text": [],  # all_text,
                "lengths": motion_lengths.numpy(),  # np.array([max_frames] * len(generated_motions)), # all_lengths,
                "num_samples": num_samples_limit,
                "num_repetitions": 1,
            },
        )
        # Noisy motion
        npy_path = os.path.join(save_dir, "results_noisy.npy")
        all_motions = target_motions.permute(0, 2, 3, 1).detach().cpu().numpy()
        print(f"saving results file to [{npy_path}]")
        np.save(
            npy_path,
            {
                "motion": all_motions,
                "text": [],  # all_text,
                "lengths": motion_lengths.numpy(),  # np.array([max_frames] * len(motion_before_edit)), # all_lengths,
                "num_samples": num_samples_limit,
                "num_repetitions": 1,
            },
        )
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
        gt_filenames = [
            os.path.join(save_dir, gt_motion_file(i)) for i in idx_to_generate
        ]
        target_filenames = [
            os.path.join(save_dir, target_motion_file(i)) for i in idx_to_generate
        ]
        gen_filenames = [
            os.path.join(save_dir, gen_motion_file(i)) for i in idx_to_generate
        ]
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
                data_rep, ground_positions, positions, l_velocity = process_file(
                    target_dup, 0.002
                )
                # data_rep is correct but need to scale with the same mean and variance
                data_rep = (data_rep - dataset.mean) / dataset.std  # [bs, 1, 196, 263]
                data_list.append(torch.from_numpy(data_rep).permute(1, 0).unsqueeze(1))
            target_rep = torch.stack(data_list, dim=0)
            # [batch_size, 263, 1, 196]
            return target_rep

        gt_motion = load_all_motion_from_filenames(gt_filenames)
        # [batch_size, 196, 22, 3]
        target_motion = load_all_motion_from_filenames(target_filenames)
        target_motion = convert_skel_to_motion_rep(target_motion, full_dataset)
        gen_motion = load_all_motion_from_filenames(gen_filenames)

        def get_motion_length():
            idx_dataset = TensorDataset(torch.tensor(idx_to_generate))
            idx_loader = DataLoader(
                idx_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
            )
            gen_loader = load_dataset(
                args,
                split,
                hml_mode="eval",
                only_idx=idx_to_generate,
                shuffle=False,
                drop_last=False,
            )
            assert (
                len(idx_loader.dataset) == len(gen_loader.dataset)
            ), f"idx_loader.dataset {len(idx_loader.dataset)} != gen_loader.dataset {len(gen_loader.dataset)}"
            motion_length = []
            for (idx,), (motion_rep, model_kwargs) in tqdm(zip(idx_loader, gen_loader)):
                motion_length.append(model_kwargs["y"]["lengths"])
            motion_length = torch.cat(motion_length, dim=0)
            return motion_length

        def load_differnt_to_motions(idx_to_generate):
            all_idxs = [*range(dataset_size)]
            idxs_left = list(set(all_idxs) - set(idx_to_generate))
            other_idxs = np.random.choice(
                idxs_left, size=len(idx_to_generate), replace=False
            )

            gen_loader = load_dataset(
                args,
                split,
                hml_mode="eval",
                only_idx=other_idxs,
                shuffle=False,
                drop_last=False,
            )
            motion_length = []
            motion = []
            for motion_rep, model_kwargs in tqdm(gen_loader):
                motion_length.append(model_kwargs["y"]["lengths"])
                motion.append(motion_rep)
            motion_length = torch.cat(motion_length, dim=0)
            motion = torch.cat(motion, dim=0)
            return motion, motion_length

        gt_length = get_motion_length()
        gt2_motion, gt2_length = load_differnt_to_motions(idx_to_generate)

        fid_gt_gt2 = calculate_fid_given_two_populations(
            gt_motion,
            gt2_motion,
            gt_length,
            gt2_length,
            dataset=full_dataset,
            dataset_name="humanml",
            device=device,
            batch_size=64,
        )
        fid_gt_target = calculate_fid_given_two_populations(
            gt_motion,
            target_motion,
            gt_length,
            gt_length,
            dataset=full_dataset,
            dataset_name="humanml",
            device=device,
            batch_size=64,
        )
        fid_gt_gen = calculate_fid_given_two_populations(
            gt_motion,
            gen_motion,
            gt_length,
            gt_length,
            dataset=full_dataset,
            dataset_name="humanml",
            device=device,
            batch_size=64,
        )
        return {
            "fid_gt_gt2": fid_gt_gt2,
            "fid_gt_target": fid_gt_target,
            "fid_gt_gen": fid_gt_gen,
        }

    fids = calculate_fid()
    metrics, metrics_target, metrics_gt = calculate_results(
        gt_motions, target_motions, generated_motions, motion_lengths, target_joints
    )
    with open(log_file, "w") as f:
        for name, eval_results in zip(
            ["Ground truth", "Target", "Prediction"],
            [metrics_gt, metrics_target, metrics],
        ):
            print(f"==================== {name} ====================")
            print(
                f"==================== {name} ====================", file=f, flush=True
            )
            for metric_name, metric_values in eval_results.items():
                metric_values = np.array(metric_values)
                unit_name = ""
                if metric_name == "Jitter":
                    unit_name = "(m/s^3)"
                elif metric_name == "MPJPE observed" or metric_name == "MPJPE all":
                    unit_name = "(m)"
                elif metric_name == "Foot skating":
                    unit_name = "(ratio)"
                print(
                    f"Metric [{metric_name} {unit_name}]: Mean {metric_values.mean():.4f}, Std {metric_values.std():.4f}"
                )
                print(
                    f"Metric [{metric_name} {unit_name}]: Mean {metric_values.mean():.4f}, Std {metric_values.std():.4f}",
                    file=f,
                    flush=True,
                )
        for fid_name, fid_value in fids.items():
            print(f"FID [{fid_name}]: {fid_value:.4f}")
            print(f"FID [{fid_name}]: {fid_value:.4f}", file=f, flush=True)


def run(
    noise_opt_conf,
    num_samples_limit=10,
    batch_size=10,
    joints="all",
    noisy=True,
    debug=False,
    added_noise_level=0.01,
):
    noise_opt_conf.postfix += f"_bs{batch_size}"
    #### Noise Optimization Config ####
    # Select task to evaluate
    task = "motion_projection"
    ##################################

    args = eval_args()
    # args = evaluation_parser()
    args.batch_size = batch_size
    args.num_frames = 196  # This must be 196!

    args.eval_project_num_joints = joints
    args.eval_project_noisy_target = noisy

    print(f"args: {args}")

    if args.eval_project_num_joints != "all":
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
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")
    # save_dir = os.path.join(os.path.dirname(args.model_path), f'eval_{task}_{niter}_opt{opt_steps}_unroll{noise_opt_conf.unroll_steps}')

    if args.eval_project_num_joints == "all":
        # for compatibility reasons
        save_dir = os.path.join(
            os.path.dirname(args.model_path), f"eval_{task}_{noise_opt_conf.name}"
        )
    else:
        save_dir = os.path.join(
            os.path.dirname(args.model_path), f"eval_{task}", noise_opt_conf.name
        )

    log_file = os.path.join(save_dir, f"eval_N{num_samples_limit}.txt")
    print("> Saving the generated motion to {}".format(save_dir))

    # if args.guidance_param != 1.:
    #     log_file += f'_gscale{args.guidance_param}'
    print(f"> Will save to log file [{log_file}]")

    print(f"> Eval task [{task}]")
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


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_num_threads(1)

    lr = 5e-2  # 5e-2 (adam), 0.02 (sgd)

    # no decorr, compare sign, unit, normal
    for num_limits in [5]:
        for (
            grad_mode,
            decorr,
            std_af,
            warm_up,
            scheduler,
            steps,
            perturb,
            sep_ode,
            unroll,
        ) in [
            # debug setting
            # ("unit", 1e3, False, 0, "cosine", 50, 0, False, 10),
            # # the best setting
            ("unit", 1e3, False, 50, "cosine", 500, 0, False, 10),
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
                ("all", True, 0.05),
                # ('three', True, 0.05),
                # ('five', True, 0.05),
                # ('lower', True, 0.05),
                # new joints
                # ('pfive', True, 0.05),
                # ('eight', True, 0.05),
                # ('ten', True, 0.05),
                # ('pfive', False, 0.05),
                # ('eight', False, 0.05),
                # ('ten', False, 0.05),
            ]:
                noise_opt_conf = NoiseOptOptions(
                    unroll_steps=unroll,  # 10,  #
                    opt_steps=steps,  #
                    optimizer="adam",  # adam
                    grad_mode=grad_mode,  # sign, unit
                    lr=lr,
                    perturb_scale=perturb,
                    diff_penalty_scale=0,
                    decorrelate_scale=decorr,  # 2e6, 1e6 (for split grad), 100 (for combined grad),  1e3 (for unit grad, split)
                    standardize_z_after_step=std_af,
                    separate_backward_for_ode=sep_ode,
                    lr_warm_up_steps=warm_up,
                    lr_scheduler=scheduler,
                    lr_decay_steps=steps,
                    postfix="test",
                )
                run(
                    noise_opt_conf,
                    num_samples_limit=num_limits,
                    batch_size=5,  #  batch_size=30,
                    joints=task,
                    noisy=noisy,
                    added_noise_level=added_noise,
                    debug=True,
                )
