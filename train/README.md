# Multi-Context Attention for Human Pose Estimation  (Training Code)

This repository includes Torch code for training of the network presented in:

> Xiao Chu, Wei Yang, Wanli Ouyang, Cheng Ma, Alan L. Yuille, Xiaogang Wang,
> **Multi-Context Attention for Human Pose Estimation**,
> CVPR, 2017. ([arXiv preprint](https://arxiv.org/abs/1702.07432))

The code is developed upon [Stacked Hourglass Network](https://github.com/anewell/pose-hg-train).

## Prerequisite
This model needs 4x GPUs with 12 GB memories. We will release the optimized model with less memory consumption soon. Stay tuned!

## Usage

1. Replace the `images` directory under `../data/mpii/` by `images` from the MPII dataset.
2. Go to `./exp` directory.
3. Run `sh train.sh`

### Note
- The training log (`train.log` and `valid.log`) and the models will be saved into `../checkpoints` directory.

## Tips
- We train our model for 130 epochs, and manually drop the learning rate by a factor of 10 at the 100 epoch and the 125 epoch.  
- [Itersize](https://github.com/gcr/torch-residual-networks/blob/master/train-imagenet-small-singleGPU-BROKEN.lua) technique can be used if you only have a limited computing power. 
- Contributions to this repo is welcomed. 