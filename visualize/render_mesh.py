import argparse
import os
from visualize import vis_utils
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help='stick figure mp4 file to be rendered.')
    parser.add_argument("--cuda", type=bool, default=True, help='')
    parser.add_argument("--device", type=int, default=0, help='')
    parser.add_argument("--sample", type=int, default=0, help='')
    params = parser.parse_args()

    task = "normal"
    # task = "motion_refinement"
    # task = "motion_editing"

    if task == "normal":
        assert params.input_path.endswith('.mp4')
        parsed_name = os.path.basename(params.input_path).replace('.mp4', '').replace('sample', '').replace('rep', '')
        sample_i, rep_i = [int(e) for e in parsed_name.split('_')]
        npy_path = os.path.join(os.path.dirname(params.input_path), 'results.npy')
        out_npy_path = params.input_path.replace('.mp4', '_smpl_params.npy')
        
        assert os.path.exists(npy_path)
        results_dir = params.input_path.replace('.mp4', '_obj')

        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)
        os.makedirs(os.path.join(results_dir, "loc"))

        npy2obj = vis_utils.npy2obj(npy_path, sample_i, rep_i,
                                    device=params.device, cuda=params.cuda)

        print('Saving obj files to [{}]'.format(os.path.abspath(results_dir)))
        for frame_i in tqdm(range(npy2obj.real_num_frames)):
            npy2obj.save_ply(os.path.join(results_dir, 'frame{:03d}.ply'.format(frame_i)), frame_i)

        print("merge ply to npy for mesh rendering")
        vis_utils.plys2npy(results_dir, results_dir)


    elif task == "motion_refinement":
        for if_noisy in [True, False]:
            for ii in range(20):
                motion_id = ii # 0
                sample_i = ii
                rep_i = 0
                NOISY_INPUT = if_noisy  # True
                if NOISY_INPUT:
                    bf = "_noisy"
                else:
                    bf = ""

                base_path = os.path.dirname(params.input_path)
                npy_path = os.path.join(base_path, "results" + f"{bf}" +".npy")
                out_npy_path = os.path.join(base_path, f"{motion_id}" + f"{bf}" + "_smpl_params.npy")
                results_dir = os.path.join(base_path, f"{motion_id}" + f"{bf}" + "_obj")

                if os.path.exists(results_dir):
                    shutil.rmtree(results_dir)
                os.makedirs(results_dir)
                os.makedirs(os.path.join(results_dir, "loc"))

                npy2obj = vis_utils.npy2obj(npy_path, sample_i, rep_i,
                                            device=params.device, cuda=params.cuda)

                print('Saving obj files to [{}]'.format(os.path.abspath(results_dir)))
                # import pdb; pdb.set_trace()
                for frame_i in tqdm(range(npy2obj.real_num_frames)):
                    npy2obj.save_ply(os.path.join(results_dir, 'frame{:03d}.ply'.format(frame_i)), frame_i)

                print("merge ply to npy for mesh rendering")
                vis_utils.plys2npy(results_dir, results_dir)

    elif task == "motion_editing":
        # BEFORE_EDIT = True
        BEFORE_EDIT = False
        # For editing task #################
        for ii in range(0, 11):

            motion_id = ii # 0
            sample_i = ii # 0
            rep_i = 0 # ii
            if BEFORE_EDIT:
                bf = "_before_edit"
            else:
                bf = ""

            base_path = os.path.dirname(params.input_path)
            npy_path = os.path.join(base_path, "results" + f"{bf}" +".npy")
            out_npy_path = os.path.join(base_path, f"{motion_id}" + f"{bf}" + "_smpl_params.npy")
            results_dir = os.path.join(base_path, f"{motion_id}" + f"{bf}" + "_obj")

            if os.path.exists(results_dir):
                shutil.rmtree(results_dir)
            os.makedirs(results_dir)
            os.makedirs(os.path.join(results_dir, "loc"))

            npy2obj = vis_utils.npy2obj(npy_path, sample_i, rep_i,
                                        device=params.device, cuda=params.cuda)

            print('Saving obj files to [{}]'.format(os.path.abspath(results_dir)))
            for frame_i in tqdm(range(npy2obj.real_num_frames)):
                npy2obj.save_ply(os.path.join(results_dir, 'frame{:03d}.ply'.format(frame_i)), frame_i)


            print("merge ply to npy for mesh rendering")
            vis_utils.plys2npy(results_dir, results_dir)


