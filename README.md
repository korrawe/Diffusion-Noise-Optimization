# Diffusion-Noise-Optimization
DNO: Optimizing Diffusion Noise Can Serve As Universal Motion Priors

[![arXiv](https://img.shields.io/badge/arXiv-<2312.11994>-<COLOR>.svg)](https://arxiv.org/abs/2312.11994)

The official PyTorch implementation of the paper [**"DNO: Optimizing Diffusion Noise Can Serve As Universal Motion Priors"**](https://arxiv.org/abs/).

Visit our [**project page**](https://korrawe.github.io/dno-project/) for more details.

![teaser](./assets/teaser.jpg)


#### Bibtex
If you find this code useful in your research, please cite:

```
@inproceedings{karunratanakul2023dno,
  title     = {Optimizing Diffusion Noise Can Serve As Universal Motion Priors},
  author    = {Karunratanakul, Korrawe and Preechakul, Konpat and Aksan, Emre and Beeler, Thabo and Suwajanakorn, Supasorn and Tang, Siyu},
  booktitle = {arxiv:2312.11994},
  year      = {2023}
}
```

## News

📢 **9/May/24** - Initial release with functional generation and evaluation code. This is an early release so please expect some undocumented piece of code.

Visualization code and detailed comments will follow.


## Getting started

**Important**: DNO is model agnostic and can be used with any diffusion model. The main file is `dno.py`. The demo code for different tasks is in `sample/gen_dno.py`.

This demo will show the result using MDM with Exponential Model Averaging (EMA) which we trained ourselves.


The environment setup is the same as [GMD](https://github.com/korrawe/guided-motion-diffusion). If you already have a working environment, it should also work here.

This code was tested on `Ubuntu 20.04 LTS` and requires:

* Python 3.7
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### 1. Setup environment

Install ffmpeg (if not already installed):

```shell
sudo apt update
sudo apt install ffmpeg
```
For windows use [this](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/) instead.


### 2. Install dependencies

DNO uses the same dependencies as GMD so if you already install one, you can use the same environment here.

Setup conda env:

```shell
conda env create -f environment_gmd.yml
conda activate gmd
conda remove --force ffmpeg
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```

Download dependencies:


<summary><b>Text to Motion</b></summary>

```bash
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```



### 2. Get data

<!-- <details>
  <summary><b>Text to Motion</b></summary> -->

There are two paths to get the data:

(a) **Generation only** wtih pretrained text-to-motion model without training or evaluating

(b) **Get full data** to train and evaluate the model.


#### a. Generation only (text only)

**HumanML3D** - Clone HumanML3D, then copy the data dir to our repository:

```shell
cd ..
git clone https://github.com/EricGuo5513/HumanML3D.git
unzip ./HumanML3D/HumanML3D/texts.zip -d ./HumanML3D/HumanML3D/
cp -r HumanML3D/HumanML3D motion-diffusion-model/dataset/HumanML3D
cd motion-diffusion-model
cp -a dataset/HumanML3D_abs/. dataset/HumanML3D/
```


#### b. Full data (text + motion capture)


**HumanML3D** - Follow the instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git),
then copy the result dataset to our repository:



Then copy the data to our repository
```shell
cp -r ../HumanML3D/HumanML3D ./dataset/HumanML3D
```

### 3. Download the pretrained models

Download our version of MDM, then unzip and place it in `./save/`. 
The model is trained on the HumanML3D dataset.

[MDM model with EMA](https://polybox.ethz.ch/index.php/s/ZiXkIzdIsspK2Lt)


## Motion Synthesis
We provide a demo code for motion editing, in-filling, refinement, in-beetweening, and blending tasks in `sample/gen_dno.py`.
The task can be selected by commenting or uncommenting from list on lines 54-58.

**Note**: The only differences between these tasks are the reward/loss function and whether to start from DDIM inverted noise or random noise. The rest of the framework is the same.

The demo targets are currently hardcoded in `sample/dno_helper.py` and can be modified to your own target (your own function or hardcoded target pose/location). 
In all tasks, the target pose and the mask need to be specified.

```shell
python -m sample.gen_dno --model_path ./save/mdm_avg_dno/model000500000_avg.pt --text_prompt "a person is jumping forward"
```

### Motion Editing
For motion editing there is a UI for trajectory editing that can be used with the flag `USE_GUI` as follows: 
- Slide the bar to select frame.
- Click the location you want to edit to.
- Click add.
- Repeat until you are satisfy then click done.


## Useful Notes
- We use 500-800 iterations for the optimization with 10 DDIM steps. The number of iterations can be adjusted with the `num_opt_steps` in `DNOOption` and `num_ode_steps` in `gen_dno.py`.
- More iterations can lead to better results but also longer computation time. DDIM steps can be increased as well.
- We found that more DDIM steps require more iterations to converge (1000-2000 steps) but generally produce better results.


## Evaluation
We provide a script to evaluate the refinement task. The script will evaluate the model on the HumanML3D dataset by adding noise to the ground truth motion.
```shell
python -m eval.eval_refinement --model_path ./save/mdm_avg_dno/model000500000_avg.pt
```

## Visualization
To be updated.


## Acknowledgments

We would like to thank the following contributors for the great foundation that we build upon:

[GMD](https://github.com/korrawe/guided-motion-diffusion),
[MDM](https://github.com/GuyTevet/motion-diffusion-model), [guided-diffusion](https://github.com/openai/guided-diffusion), [MotionCLIP](https://github.com/GuyTevet/MotionCLIP), [text-to-motion](https://github.com/EricGuo5513/text-to-motion), [actor](https://github.com/Mathux/ACTOR), [joints2smpl](https://github.com/wangsen1312/joints2smpl), [MoDi](https://github.com/sigal-raab/MoDi).

## License
This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including CLIP, SMPL, SMPL-X, PyTorch3D, and uses datasets that each have their own respective licenses that must also be followed.
