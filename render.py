import os
import random
import shutil
import sys
from pathlib import Path

import natsort
import pickle
import numpy as np

try:
    import bpy

    sys.path.append(os.path.dirname(bpy.data.filepath))
except ImportError:
    raise ImportError(
        "Blender is not properly installed or not launch properly. See README.md to have instruction on how to install and use blender."
    )

import mld.launch.blender
import mld.launch.prepare  # noqa
from mld.config import parse_args
from mld.utils.joints import smplh_to_mmm_scaling_factor


def extend_paths(path, keyids, *, onesample=True, number_of_samples=1):
    if not onesample:
        template_path = str(path / "KEYID_INDEX.npy")
        paths = [
            template_path.replace("INDEX", str(index)) for i in range(number_of_samples)
        ]
    else:
        paths = [str(path / "KEYID.npy")]

    all_paths = []
    for path in paths:
        all_paths.extend([path.replace("KEYID", keyid) for keyid in keyids])
    return all_paths


def render_cli() -> None:
    # parse options
    cfg = parse_args(phase="render")  # parse config file
    cfg.FOLDER = cfg.RENDER.FOLDER
    # output_dir = Path(os.path.join(cfg.FOLDER, str(cfg.model.model_type) , str(cfg.NAME)))
    # create logger
    # logger = create_logger(cfg, phase='render')

    if cfg.RENDER.INPUT_MODE.lower() == "npy":
        output_dir = Path(os.path.dirname(cfg.RENDER.NPY))
        paths = [cfg.RENDER.NPY]
        # print("xxx")
        # print("begin to render for{paths[0]}")
    elif cfg.RENDER.INPUT_MODE.lower() == "dir":
        output_dir = Path(cfg.RENDER.DIR)
        paths = []
        # file_list = os.listdir(cfg.RENDER.DIR)
        # random begin for parallel
        file_list = natsort.natsorted(os.listdir(cfg.RENDER.DIR))
        begin_id = random.randrange(0, len(file_list))
        file_list = file_list[begin_id:]+file_list[:begin_id]

        
        # render mesh npy first
        for item in file_list:
            if item.endswith("_mesh.npy"):
                paths.append(os.path.join(cfg.RENDER.DIR, item))

        # then render other npy
        for item in file_list:
            if item.endswith(".npy") and not item.endswith("_mesh.npy"):
                paths.append(os.path.join(cfg.RENDER.DIR, item))        
        
        # additional_objects = {
        #     "drag": ((0., 0., 0.), (2, 2., 0.)), # Start - end
        #     "obs": (1., -1., 0., 0.5), # x, y, z, radius
        #     "target": (0., 0., 0.), # x, y, z
        # }
        # # Save additional_objects to a pickle file
        # with open(pickle_file, "wb") as f:
        #     pickle.dump(additional_objects, f)

        # EDITED_SEQ = True
        EDITED_SEQ = False
        # EDITED_SEQ = (True if ("sample02_rep06_obj" in paths[0] or "sample04" in paths[0] or "sample03" in paths[0]) else False)
        # EDITED_SEQ = (False if "sample04_rep06_obj" in paths[0] else True)
        SYNZED_ADDITIONAL_DATA = False # True # False
        additional_objects = None
        # LOAD_OBJECTIVES = True # False # True
        LOAD_OBJECTIVES = False
        if LOAD_OBJECTIVES:
            # Load additional data
            pickle_file = os.path.join(os.path.dirname(cfg.RENDER.DIR), "additional_objects.pkl")
            # load additional_objects
            additional_objects = pickle.load(open(pickle_file, "rb"))
        
        LOAD_DRAG_TRAJ = True
        if LOAD_DRAG_TRAJ:
            # Load drag traj
            pickle_file = os.path.join(os.path.dirname(cfg.RENDER.DIR), "edit_trajectories.pkl")
            # load additional_objects
            drag_traj = pickle.load(open(pickle_file, "rb"))
            seq = os.path.basename(paths[0]).split("_")[0]
            seq = int(seq)
            additional_objects = {
                "drag": [(drag_traj["start"][seq], drag_traj["end"][seq])], # Start - end
                # "obs": (1., -1., 0., 0.5), # x, y, z, radius
                # "target": (0., 0., 0.), # x, y, z
            }
            # import pdb; pdb.set_trace()
            # additional_objects = drag_traj
        
        EDIT_COMBINE = True # False
        if EDIT_COMBINE:
            seq_list = [1, 2, 3, 4, 5, 6, 7, 9]
            # Load all drag traj
            pickle_file = os.path.join(os.path.dirname(cfg.RENDER.DIR), "edit_trajectories.pkl")
            drag_traj = pickle.load(open(pickle_file, "rb"))
            # seq = os.path.basename(paths[0]).split("_")[0]
            # seq = int(seq)
            additional_objects = {}
            drag_list = []
            for seq in seq_list:
                drag_list.append((drag_traj["start"][seq], drag_traj["end"][seq])) # (Start, End)
                # "drag": [(drag_traj["start"][seq], drag_traj["end"][seq])], 
            additional_objects["drag"] = drag_list

            # Load list of motion


        LOAD_NOISY_DATA = False # True
        # LOAD_NOISY_DATA = False
        noisy_inputs = None
        if LOAD_NOISY_DATA:
            # Load noisy data
            npy_path = os.path.join(os.path.dirname(cfg.RENDER.DIR), "results_noisy.npy")
            # npy_path = os.path.join(os.path.dirname(cfg.RENDER.DIR), "results.npy")
            # load additional_objects
            noisy_inputs_file = np.load(npy_path, allow_pickle=True)
            noisy_inputs_file = noisy_inputs_file[None][0]
            noisy_inputs = noisy_inputs_file['motion']
            seq = os.path.basename(paths[0]).split("_")[0]
            seq = int(seq)
            noisy_inputs = noisy_inputs[seq]
            # import pdb; pdb.set_trace()

        print(f"begin to render for {paths[0]}")

    # import numpy as np

    from mld.render.blender import render
    from mld.render.blender.tools import mesh_detect
    from mld.render.video import Video
    init = True
    for path in paths:
        
        # check existed mp4 or under rendering
        if cfg.RENDER.MODE == "video":
            if os.path.exists(path.replace(".npy", ".mp4")) or os.path.exists(path.replace(".npy", "_frames")):
                print(f"npy is rendered or under rendering {path}")
                continue
        else:
            # check existed png
            if os.path.exists(path.replace(".npy", ".png")):
                print(f"npy is rendered or under rendering {path}")
                continue
        
        if cfg.RENDER.MODE == "video":
            frames_folder = os.path.join(
                output_dir, path.replace(".npy", "_frames").split('/')[-1])
            os.makedirs(frames_folder, exist_ok=True)
        else:
            frames_folder = os.path.join(
                output_dir, path.replace(".npy", ".png").split('/')[-1])

        try:
            data = np.load(path)
            if cfg.RENDER.JOINT_TYPE.lower() == "humanml3d":
                is_mesh = mesh_detect(data)
                if not is_mesh:
                    data = data * smplh_to_mmm_scaling_factor
            # import pdb; pdb.set_trace()
            # import trimesh
            # outout = "/home/korrawe/motion_gen/motion-diffusion-model/save/gmd_final/samples_000500000_seed10_a_person_is_walking/test_mesh_110.ply"
            # # maybe the face files is the problem?
            # faces = np.load(cfg.RENDER.FACES_PATH)
            # mesh_out = trimesh.Trimesh(vertices=data[110], faces=faces)
            # mesh_out.export(outout)
            # import pdb; pdb.set_trace()
        except FileNotFoundError:
            print(f"{path} not found")
            continue

        
        add_data = None
        if SYNZED_ADDITIONAL_DATA:
            if EDITED_SEQ:
                add_mesh_name = "sample00_rep06_obj"
            else:
                add_mesh_name = "sample02_rep06_obj"
                # add_mesh_name = "sample03_rep06_obj"
                # add_mesh_name = "sample04_rep06_obj"
            add_data_path = os.path.join(os.path.dirname(output_dir), add_mesh_name ,add_mesh_name + "_mesh.npy")
            add_data = np.load(add_data_path)
        if EDIT_COMBINE:
            # Load multiple mesh sequence to render them at once
            add_data = []
            for seq in seq_list:
                add_mesh_name = f"{str(seq)}_obj"
                add_data_i_path = os.path.join(os.path.dirname(output_dir), add_mesh_name ,add_mesh_name + "_mesh.npy")
                add_data_i = np.load(add_data_i_path)
                add_data.append(add_data_i)

        if cfg.RENDER.MODE == "video":
            frames_folder = os.path.join(
                output_dir, path.replace(".npy", "_frames").split("/")[-1]
            )
        else:
            frames_folder = os.path.join(
                output_dir, path.replace(".npy", ".png").split("/")[-1]
            )

        # Move body on floor
        # cfg.RENDER.ALWAYS_ON_FLOOR = True

        out = render(
            data,
            frames_folder,
            canonicalize=cfg.RENDER.CANONICALIZE,
            exact_frame=cfg.RENDER.EXACT_FRAME,
            num=cfg.RENDER.NUM,
            mode=cfg.RENDER.MODE,
            faces_path=cfg.RENDER.FACES_PATH,
            downsample=cfg.RENDER.DOWNSAMPLE,
            always_on_floor=cfg.RENDER.ALWAYS_ON_FLOOR,
            oldrender=cfg.RENDER.OLDRENDER,
            jointstype=cfg.RENDER.JOINT_TYPE.lower(),
            res=cfg.RENDER.RES,
            init=init,
            gt=not EDITED_SEQ, # cfg.RENDER.GT,
            accelerator=cfg.ACCELERATOR,
            device=cfg.DEVICE,
            additional_objects=additional_objects,
            npy_add_data=add_data,
            edited_seq=EDITED_SEQ,
            noisy_inputs=noisy_inputs,
        )

        init = False

        if cfg.RENDER.MODE == "video":
            if cfg.RENDER.DOWNSAMPLE:
                video = Video(frames_folder, fps=cfg.RENDER.FPS)
            else:
                video = Video(frames_folder, fps=cfg.RENDER.FPS)

            vid_path = frames_folder.replace("_frames", ".mp4")
            video.save(out_path=vid_path)
            shutil.rmtree(frames_folder)
            print(f"remove tmp fig folder and save video in {vid_path}")

        else:
            print(f"Frame generated at: {out}")


if __name__ == "__main__":
    render_cli()
