import os.path as osp

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_loaders.humanml.data.dataset import sample_to_motion
from data_loaders.humanml.scripts.motion_process import process_file
from data_loaders.humanml.utils.metrics import *
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.humanml.utils.utils import *
from diffusion import logger
from diffusion.gaussian_diffusion import GaussianDiffusion
from dno import DNO, DNOOptions
from eval.calculate_fid import calculate_fid_given_two_populations
from eval.eval_shared import *
from model.cfg_sampler import ClassifierFreeSampleModel
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import (
    create_gaussian_diffusion,
    create_model_and_diffusion,
    load_model_wo_clip,
)
from utils.parser_util import evaluation_parser


def evaluation(
    dno_conf: DNOOptions,
    num_samples_limit,
    joints,
    batch_size,
    added_noise_level=0.05,
    num_ode_steps=10,
    noisy=True,
    debug=False,
    use_gradient_checkpoint=False,
):
    #### Noise Optimization Config ####
    # Select task to evaluate
    task = "motion_projection_dno"
    ##################################

    args = evaluation_parser()
    args.batch_size = batch_size
    # for dataset (must be 196)
    args.num_frames = 196

    print(f"args: {args}")

    #########################
    # task names and save paths
    if joints != "all":
        task += f"_{joints}"
    if not noisy:
        task += "_clean"
    else:
        print(f"Added noise level: {added_noise_level}")
        if added_noise_level != 0.01:
            task += f"_noise{added_noise_level}"
    if use_gradient_checkpoint:
        task += "_gc"

    name = os.path.basename(os.path.dirname(args.model_path))
    save_dir = os.path.join(os.path.dirname(args.model_path), f"eval_{task}")
    log_file = os.path.join(save_dir, f"eval_N{num_samples_limit}.txt")
    os.makedirs(save_dir, exist_ok=True)
    print("> Saving the generated motion to {}".format(save_dir))
    print(f"> Will save to log file [{log_file}]")
    print(f"> Eval task [{task}]")

    ### Task settings ###
    print(f"args: {args}")
    print(f"noisy: {noisy}")
    joint_to_condition = joints  # 'all' or 'upper' or 'lower' or 'three' or 'five'
    print(f"joint_to_condition: {joint_to_condition}")
    target_joints = get_target_joints(joint_to_condition)
    n_frames = 196
    device = "cuda"
    #####################

    # fix the global seeds
    fixseed(args.seed)

    # fixed the random indexes
    np.random.seed(0)
    split = "test"
    eval_dataset = load_dataset(args, split, hml_mode="eval", with_loader=False)
    eval_dataset_size = len(eval_dataset)
    print(f"dataset_size: {eval_dataset_size}")
    # random the indexes to generate from the dataset
    eval_idx_to_generate = np.random.choice(
        eval_dataset_size, size=num_samples_limit, replace=False
    )
    print(f"id_to_generate: {eval_idx_to_generate}")

    # finding the indexes that are not in the cache (that need to be generated)
    eval_idx_not_in_cache = []

    def gen_motion_file(i):
        return f"motion_{i:05d}.pt"

    def target_motion_file(i):
        return f"target_{i:05d}.pt"

    def gt_motion_file(i):
        return f"gt_{i:05d}.pt"

    def hist_motion_file(i):
        return f"hist_{i:05d}.pt"

    for i in eval_idx_to_generate:
        motion_path = os.path.join(save_dir, gen_motion_file(i))
        if not os.path.exists(motion_path):
            eval_idx_not_in_cache.append(i)
        else:
            eval_idx_not_in_cache.append(i)  # to remove
            print(f"Motion [{motion_path}] already exists, skipping...")
    # create a dataloader with only the indexes that need to be generated
    eval_idx_dataset = TensorDataset(torch.tensor(eval_idx_not_in_cache))
    eval_idx_loader = DataLoader(
        eval_idx_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    # create a motion dataset with only the indexes that need to be generated
    eval_motion_loader = load_dataset(
        args,
        split,
        hml_mode="eval",
        only_idx=eval_idx_not_in_cache,
        shuffle=False,
        drop_last=False,
    )
    assert (
        len(eval_idx_loader.dataset) == len(eval_motion_loader.dataset)
    ), f"idx_loader.dataset {len(eval_idx_loader.dataset)} != gen_loader.dataset {len(eval_motion_loader.dataset)}"

    # loading the model
    logger.log("Creating model and diffusion...")
    model, _ = create_model_and_diffusion(args, eval_motion_loader)
    diffusion = create_gaussian_diffusion(
        args, timestep_respacing=f"ddim{num_ode_steps}"
    )

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(
            model
        )  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    ###########################################
    # Generate motions via noise optimization
    for (idx,), (motion_rep, model_kwargs) in tqdm(
        zip(eval_idx_loader, eval_motion_loader)
    ):
        # idx [bs, ]
        # motion [batch_size, 263, 1, 196]
        batch_size = len(motion_rep)
        gt_skels = input_to_motion(motion_rep, eval_motion_loader, model)

        # optimization target
        # motion refinement = noisy target
        # motion completion = clean target
        target = gt_skels
        if noisy:
            target = target + torch.randn_like(target) * added_noise_level
        # set mask to match target lengths
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

        # Optimize here without the text prompt
        model_kwargs["y"]["text"] = [""]
        # add CFG scale to batch
        if args.guidance_param != 1.0:
            model_kwargs["y"]["scale"] = (
                torch.ones(batch_size, device=dist_util.dev()) * args.guidance_param
            )

        # initialize noise
        gen_shape = [batch_size, model.njoints, model.nfeats, n_frames]
        cur_xt = torch.randn(gen_shape).to(device)

        def criterion(x):
            motion_length = model_kwargs["y"]["lengths"]
            inv_transform = eval_motion_loader.dataset.t2m_dataset.inv_transform_th
            n_joints = 22

            # motion_mask shape [bs, 120, 22, 3]
            if motion_length is None:
                motion_mask = torch.ones_like(target_mask)
            else:
                # the mask is only for the first motion_length frames
                motion_mask = torch.zeros_like(target_mask)
                for i, m_len in enumerate(motion_length):
                    motion_mask[i, :m_len, :, :] = 1.0

            x_in_pose_space = inv_transform(
                x.permute(0, 2, 3, 1),
                use_rand_proj=False,
            )  # [bs, 1, 120, 263]
            # Compute (x,y,z) shape [bs, 1, 120, njoints=22, nfeat=3]
            x_in_joints = recover_from_ric(x_in_pose_space, n_joints, abs_3d=False)
            # remove the feature dim
            x_in_joints = x_in_joints.squeeze(1)

            # l1 works better than l2 loss
            loss = (
                F.l1_loss(x_in_joints, target, reduction="none")
                * target_mask
                * motion_mask
            )
            # average the loss over the number of valid joints
            loss = loss.sum(dim=[1, 2, 3]) / (target_mask * motion_mask).sum(
                dim=[1, 2, 3]
            )
            return loss

        def solver(z):
            return ddim_loop_with_gradient(
                diffusion,
                model,
                (batch_size, model.njoints, model.nfeats, n_frames),
                model_kwargs=model_kwargs,
                noise=z,
                gradient_checkpoint=use_gradient_checkpoint,
                clip_denoised=False,
            )

        # start optimizing
        noise_opt = DNO(
            model=solver, criterion=criterion, start_z=cur_xt, conf=dno_conf
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

    #############################################
    # load the generated motions for evaluations
    generated_motions, target_motions, gt_motions = [], [], []
    motion_lengths = []
    # load the generated motions from files
    eval_idx_dataset = TensorDataset(torch.tensor(eval_idx_to_generate))
    eval_idx_loader = DataLoader(
        eval_idx_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    eval_motion_loader = load_dataset(
        args,
        split,
        hml_mode="eval",
        only_idx=eval_idx_to_generate,
        shuffle=False,
        drop_last=False,
    )
    assert (
        len(eval_idx_loader.dataset) == len(eval_motion_loader.dataset)
    ), f"idx_loader.dataset {len(eval_idx_loader.dataset)} != gen_loader.dataset {len(eval_motion_loader.dataset)}"
    for (idx,), (motion_rep, model_kwargs) in tqdm(
        zip(eval_idx_loader, eval_motion_loader)
    ):
        # save the gt file (for reference)
        for i in range(len(motion_rep)):
            gt_file = os.path.join(save_dir, gt_motion_file(idx[i]))
            # if gt file doesn't exist create one and save
            if not os.path.exists(gt_file):
                torch.save(motion_rep[i].detach().cpu().clone(), gt_file)

        gt_skels = input_to_motion(motion_rep, eval_motion_loader, model)
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
                generated.unsqueeze(0), eval_motion_loader.dataset, model, abs_3d=False
            )
            generated = generated.permute(0, 3, 1, 2)
            generated_motions.append(generated)

            # plot the optimization stats
            if debug:
                hist = torch.load(os.path.join(save_dir, hist_motion_file(i)))
                for key in [
                    "loss",
                    "loss_diff",
                    "loss_decorrelate",
                    "grad_norm",
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
    # save motions as video
    if debug:
        for ii in range(len(generated_motions)):
            # only save the first 20 motions as video
            if ii > 20:
                break
            motion_id = f"{ii:05d}"
            plot_debug(
                generated_motions[ii],
                osp.join(save_dir, f"{motion_id}_gen.mp4"),
                eval_motion_loader,
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
                eval_motion_loader,
                motion_lengths[ii],
                clean_root=clean_root,
            )
            plot_debug(
                gt_motions[ii],
                osp.join(save_dir, f"{motion_id}_gt.mp4"),
                eval_motion_loader,
                motion_lengths[ii],
            )

    #################################
    # save results for blender renders
    SAVE_FOR_VIS = True
    if SAVE_FOR_VIS:
        # Refined motion
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

    #################################
    # calculate fid and other metrics

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
            os.path.join(save_dir, gt_motion_file(i)) for i in eval_idx_to_generate
        ]
        target_filenames = [
            os.path.join(save_dir, target_motion_file(i)) for i in eval_idx_to_generate
        ]
        gen_filenames = [
            os.path.join(save_dir, gen_motion_file(i)) for i in eval_idx_to_generate
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
        target_motion = convert_skel_to_motion_rep(target_motion, eval_dataset)
        gen_motion = load_all_motion_from_filenames(gen_filenames)

        def get_motion_length():
            idx_dataset = TensorDataset(torch.tensor(eval_idx_to_generate))
            idx_loader = DataLoader(
                idx_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
            )
            loader = load_dataset(
                args,
                split,
                hml_mode="eval",
                only_idx=eval_idx_to_generate,
                shuffle=False,
                drop_last=False,
            )
            assert (
                len(idx_loader.dataset) == len(loader.dataset)
            ), f"idx_loader.dataset {len(idx_loader.dataset)} != gen_loader.dataset {len(loader.dataset)}"
            motion_length = []
            for (idx,), (motion_rep, model_kwargs) in tqdm(zip(idx_loader, loader)):
                motion_length.append(model_kwargs["y"]["lengths"])
            motion_length = torch.cat(motion_length, dim=0)
            return motion_length

        def load_other_real_motions(idx_to_generate):
            all_idxs = [*range(eval_dataset_size)]
            idxs_left = list(set(all_idxs) - set(idx_to_generate))
            other_idxs = np.random.choice(
                idxs_left, size=len(idx_to_generate), replace=False
            )
            loader = load_dataset(
                args,
                split,
                hml_mode="eval",
                only_idx=other_idxs,
                shuffle=False,
                drop_last=False,
            )
            motion_length = []
            motion = []
            for motion_rep, model_kwargs in tqdm(loader):
                motion_length.append(model_kwargs["y"]["lengths"])
                motion.append(motion_rep)
            motion_length = torch.cat(motion_length, dim=0)
            motion = torch.cat(motion, dim=0)
            return motion, motion_length

        gt_length = get_motion_length()
        gt2_motion, gt2_length = load_other_real_motions(eval_idx_to_generate)

        fid_gt_gt2 = calculate_fid_given_two_populations(
            gt_motion,
            gt2_motion,
            gt_length,
            gt2_length,
            dataset=eval_dataset,
            dataset_name="humanml",
            device=device,
            batch_size=64,
        )
        fid_gt_target = calculate_fid_given_two_populations(
            gt_motion,
            target_motion,
            gt_length,
            gt_length,
            dataset=eval_dataset,
            dataset_name="humanml",
            device=device,
            batch_size=64,
        )
        fid_gt_gen = calculate_fid_given_two_populations(
            gt_motion,
            gen_motion,
            gt_length,
            gt_length,
            dataset=eval_dataset,
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


def ddim_loop_with_gradient(
    diffusion: GaussianDiffusion,
    model: nn.Module,
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


if __name__ == "__main__":
    torch.set_num_threads(1)

    conf = DNOOptions(num_opt_steps=500)
    evaluation(
        conf,
        num_samples_limit=300,
        joints="all",
        batch_size=16, # 30,
        noisy=True,
        added_noise_level=0.05,
        use_gradient_checkpoint=False,
        debug=True,
    )
