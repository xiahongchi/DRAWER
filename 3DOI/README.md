# Understanding 3D Object Interaction from a Single Image

Code release for our paper

**Understanding 3D Object Interaction from a Single Image**  
[Shengyi Qian][sq], [David F. Fouhey][dff]  
ICCV 2023

[[`Project Page`](https://jasonqsy.github.io/3DOI/)]  [[`arXiv`](https://arxiv.org/abs/2305.09664)] [[`demo`](https://huggingface.co/spaces/shengyi-qian/3DOI)]

![teaser](docs/teaser.png)

Please check the [project page](https://jasonqsy.github.io/3DOI/) for more details and consider citing our paper if it is helpful:

```
@inproceedings{qian2023understanding,
    title={Understanding 3D Object Interaction from a Single Image},
    author={Qian, Shengyi and Fouhey, David F},
    booktitle = {ICCV},
    year={2023}
}
```

If you are interested in the inference-only code, you can also try our demo code on [hugging face](https://huggingface.co/spaces/shengyi-qian/3DOI).


## Setup

We are using anaconda to set up the python environment. It is tested on python 3.9 and pytorch 2.0.1. pytorch3d is only required for 3D visualization.

```bash
# python
conda create -n monoarti python=3.9
conda activate monoarti

# pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# other packages
pip install accelerate
pip install submitit
pip install hydra-core --upgrade --pre
pip install hydra-submitit-launcher --upgrade
pip install pycocotools
pip install packaging plotly imageio imageio-ffmpeg matplotlib h5py opencv-python
pip install tqdm wandb visdom

# (optional, for 3D visualization) pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

Create `checkpoints` to store pretrained checkpoints.

```bash
mkdir checkpoints
```

If necessary, download our [pretrained SAM model](https://fouheylab.eecs.umich.edu/~syqian/3DOI/checkpoint_20230515.pth) and put it at `checkpoints/checkpoint_20230515.pth`.


## Dataset

The dataset is released in the [project page](https://jasonqsy.github.io/3DOI/). Please download and set the dataset root [here](https://github.com/JasonQSY/3DOI/blob/main/monoarti/monoarti/dataset.py#L19).

The dataset should be organized like this

```
- `3doi_data`
    - `3doi_v1`
    - `images`
    - `omnidata_filtered`
```

## Inference

To test the model on any 3DOI or other dataset, run

```bash
python test.py --config-name sam_inference checkpoint_path=checkpoints/checkpoint_20230515.pth output_dir=vis
```

To create video animation, run

```bash
python test.py --config-name sam_inference checkpoint_path=checkpoints/checkpoint_20230515.pth output_dir=vis test.mode='export_video'
```

## Training 

To train our model with segment-anything backbone,

```bash
python train.py --config-name sam
```

To train our model with DETR backbone,

```bash
python train.py --config-name detr
```

## Acknowledgment

We reuse the code of [ViewSeg](https://github.com/facebookresearch/viewseg), [DETR](https://github.com/facebookresearch/detr) and [Segment-Anything](https://github.com/facebookresearch/segment-anything).


[sq]: https://github.com/JasonQSY
[dff]: https://github.com/dfouhey