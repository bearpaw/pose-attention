cd ..

CUDA_VISIBLE_DEVICES=0, th main.lua -netType hg-attention -expID crf_parts -nGPU 1 -LRNKer 11 -trainBatch 1
