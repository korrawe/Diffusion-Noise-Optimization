import subprocess
import sys
from os import path

import numpy as np
import torch
from omegaconf import OmegaConf

from data_loaders.humanml.utils.plot_script import plot_3d_motion

from .generate import construct_template_variables, save_multiple_samples


def save_videos(
    num_dump_step, all_text, all_lengths, all_motions, args, kframes, captions, obs_list, target, show_target_pose
):
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
        animation_save_path = path.join(args.out_path, save_file)
        try:
            plot_3d_motion(
                animation_save_path,
                args.skeleton,
                motion,
                dataset=args.dataset,
                title=captions[sample_i],
                fps=args.fps,
                kframes=kframes,
                obs_list=obs_list,
                target_pose=target[0].cpu().numpy(),
                gt_frames=[kk for (kk, _) in kframes] if show_target_pose else [],
            )
        except subprocess.CalledProcessError as e:
            print("failed to save video")
            print("Stdout", e.stdout)

            raise e
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


if __name__ == "__main__":
    filename = sys.argv[1]
    if not path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found ...")
    res = np.load(filename, allow_pickle=True).item()
    # pprint(res)
    # print(res.keys())
    all_motions = res["motion"]
    all_text = res["text"]
    all_lengths = res["lengths"]
    num_samples = res["num_samples"]
    num_repetitions = res["num_repetitions"]
    args = OmegaConf.load(path.dirname(filename) + "/args.yml")
    args.num_samples = num_samples
    captions = [
        "Original",
    ] + [f"Prediction {i + 1}" for i in range(args.num_trials)]
    target = torch.zeros([1, args.max_frames, 22, 3]).repeat(args.num_trials, 1, 1, 1)

    save_videos(1, all_text, all_lengths, all_motions, args, [], captions, [], target, False)
