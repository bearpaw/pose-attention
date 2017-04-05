# Multi-Context Attention for Human Pose Estimation

This repository includes Torch code for evaluation and visualization of the network presented in:

> Xiao Chu, Wei Yang, Wanli Ouyang, Cheng Ma, Alan L. Yuille, Xiaogang Wang,
> **Multi-Context Attention for Human Pose Estimation**,
> CVPR, 2017. ([arXiv preprint](https://arxiv.org/abs/1702.07432))

The code is developed upon [Stacked Hourglass Network](https://github.com/anewell/pose-hg-train).

## Installation

To run this code, the following packages must be installed:

- [Torch7](https://github.com/torch/torch7)
- hdf5 (and the [torch-hdf5](https://github.com/deepmind/torch-hdf5/) package)
- cudnn
- qlua (for displaying results)
- [matio](https://github.com/soumith/matio-ffi.torch): to save predictions in Matlab's `.mat` file.

## Testing
For testing, please go to the `test` directory and follow the `README` for instructions.

## Training
For testing, please go to the `train` directory and follow the `README` for instructions.