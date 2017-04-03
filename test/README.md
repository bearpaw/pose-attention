# Multi-Context Attention for Human Pose Estimation  (Demo Code)

This repository includes Torch code for evaluation and visualization of the network presented in:

> Xiao Chu, Wei Yang, Wanli Ouyang, Cheng Ma, Alan L. Yuille, Xiaogang Wang,
> **Multi-Context Attention for Human Pose Estimation**,
> [arXiv:1702.07432](https://arxiv.org/abs/1702.07432), 2017.

The code is developed upon [Stacked Hourglass Network](https://github.com/anewell/pose-hg-demo).

## Installation

To run this code, the following packages must be installed:

- [Torch7](https://github.com/torch/torch7)
- hdf5 (and the [torch-hdf5](https://github.com/deepmind/torch-hdf5/) package)
- cudnn
- qlua (for displaying results)
- [matio](https://github.com/soumith/matio-ffi.torch): to save predictions in Matlab's `.mat` file.

## Usage

1. Download the pretrained model from [Google Drive](https://drive.google.com/open?id=0B63t5HSgY4SQZV9vN1hnMEItYTg) and save it to `exp/mpii/crf_parts/model.t7`

2. Run the demo 
`qlua main.lua demo`

3. Run on the [MPII Human Pose dataset](http://human-pose.mpi-inf.mpg.de): 
Replacing this repository's `images` directory by `images` from the MPII dataset, you can generate full predictions on the validation and test sets.
   - For generating predictions on validation set:
   `qlua main.lua predict-valid`

   - For generating predictions on testing set:
   `qlua main.lua predict-test`



### Tips
- If you do not want to visualize predictions while testing, please set `isdisplay=false` in `main.lua`.
- For multi-scale testing, you may set the range of `scale_search` in `main.lua` (*e.g.,* `local scale_search = {0.9,1.0,1.1}`). Note that it would consume more GPU memory. 
- You may add `CUDA_VISIBLE_DEVICES=0` to run on a specific GPU device (*i.e.,* GPU device with ID 0).

## Testing your own images

Please read the instructions from [Stacked Hourglass Network](https://github.com/anewell/pose-hg-demo) for details about testing your own images. 