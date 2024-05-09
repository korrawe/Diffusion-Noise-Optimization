from enum import Enum

import torch

from data_loaders.get_data import DatasetConfig, get_dataset_loader
from data_loaders.humanml.scripts.motion_process import *
from data_loaders.humanml.utils.metrics import *
from data_loaders.humanml.utils.metrics import calculate_skating_ratio


class TargetJoints(Enum):
    # TODO: remove some options
    all = "all"
    lower = "lower"
    three = "three"
    pthree = "pthree"
    five = "five"
    pfive = "pfive"
    eight = "eight"
    ten = "ten"

    @classmethod
    def from_str(cls, name: str):
        for target_joints in cls:
            if target_joints.value == name:
                return target_joints
        raise NotImplementedError(f"name [{name}] not implemented")


def get_target_joints(name: TargetJoints):
    if isinstance(name, str):
        name = TargetJoints.from_str(name)

    if name == TargetJoints.all:
        target_joints = [*range(22)]
    elif name == TargetJoints.lower:  # pelvis and knees and ankles
        target_joints = [0, 4, 5, 7, 8]
    elif name == TargetJoints.three:  # head and hands
        target_joints = [15, 20, 21]
    elif name == TargetJoints.pthree:  # adding pelvis
        target_joints = [0, 15, 20, 21]
    elif name == TargetJoints.five:  # head and hands and feet
        target_joints = [10, 11, 15, 20, 21]
    elif name == TargetJoints.pfive:  # adding pelvis
        target_joints = [0, 10, 11, 15, 20, 21]
    elif name == TargetJoints.eight:  # pelvis, head, hands, feet, shoulders
        target_joints = [0, 10, 11, 15, 16, 17, 20, 21]
    elif name == TargetJoints.ten:  # pelvis, head, hands, feet, knees, shoulders,
        target_joints = [0, 4, 5, 10, 11, 15, 16, 17, 20, 21]
    else:
        raise NotImplementedError(f"name [{name}] not implemented")
    return target_joints


def calculate_results(
    gt_motions, target_motions, generated_motions, motion_lengths, target_joints
):
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

    # Compute norm of difference between gt_motions and target_motions
    for i in range(len(gt_motions)):
        # GT
        # import pdb; pdb.set_trace()
        gt_cut = gt_motions[i, : motion_lengths[i], :, :]
        skate_ratio, _ = calculate_skating_ratio(
            gt_cut.permute(1, 2, 0).unsqueeze(0)
        )  # need input shape [bs, 22, 3, max_len]
        metrics_gt["Foot skating"].append(skate_ratio.item())
        metrics_gt["Jitter"].append(compute_jitter(gt_cut).item())

        # Target
        target_cut = target_motions[i, : motion_lengths[i], :, :]
        skate_ratio, _ = calculate_skating_ratio(
            target_cut.permute(1, 2, 0).unsqueeze(0)
        )
        metrics_target["Foot skating"].append(skate_ratio.item())
        metrics_target["Jitter"].append(compute_jitter(target_cut).item())
        metrics_target["MPJPE all"].append(
            torch.norm(gt_cut - target_cut, dim=2).mean().item()
        )
        metrics_target["MPJPE observed"].append(
            torch.norm(
                gt_cut[:, target_joints, :] - target_cut[:, target_joints, :], dim=2
            )
            .mean()
            .item()
        )

        # Generated
        gen_cut = generated_motions[i, : motion_lengths[i], :, :]
        skate_ratio, _ = calculate_skating_ratio(gen_cut.permute(1, 2, 0).unsqueeze(0))
        metrics["Foot skating"].append(skate_ratio.item())
        metrics["Jitter"].append(compute_jitter(gen_cut).item())
        metrics["MPJPE all"].append(torch.norm(gt_cut - gen_cut, dim=2).mean().item())
        metrics["MPJPE observed"].append(
            torch.norm(
                gt_cut[:, target_joints, :] - gen_cut[:, target_joints, :], dim=2
            )
            .mean()
            .item()
        )
    return metrics, metrics_target, metrics_gt


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


def load_dataset(
    args,
    split,
    hml_mode,
    only_idx=None,
    with_loader=True,
    shuffle=True,
    drop_last=True,
):
    conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        split=split,
        hml_mode=hml_mode,
    )
    data = get_dataset_loader(
        conf,
        only_idx=only_idx,
        with_loader=with_loader,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    if with_loader:
        data.fixed_length = args.num_frames
    return data


def input_to_motion(motion, gen_loader, model):
    gt_poses = motion.permute(0, 2, 3, 1)
    gt_poses = (
        gt_poses * gen_loader.dataset.std + gen_loader.dataset.mean
    )  # [bs, 1, 196, 263]
    gt_skels = recover_from_ric(gt_poses, 22, abs_3d=False).squeeze(1)
    gt_skels = gt_skels.permute(0, 2, 3, 1)
    gt_skels = model.rot2xyz(
        x=gt_skels,
        mask=None,
        pose_rep="xyz",
        glob=True,
        translation=True,
        jointstype="smpl",
        vertstrans=True,
        betas=None,
        beta=0,
        glob_rot=None,
        get_rotations_back=False,
    )
    gt_skels = gt_skels.permute(0, 3, 1, 2)
    return gt_skels
