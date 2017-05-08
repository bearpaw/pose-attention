cd ..

# Example for training with 4x 12GB GPUs
# Please refer to opts.lua for more training options
CUDA_VISIBLE_DEVICES=0,1,2,3 th main.lua -netType hg-attention -expID crf_parts -nGPU 1 -LRNKer 11 -nGPU 4
