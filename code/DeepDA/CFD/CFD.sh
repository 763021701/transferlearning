#!/usr/bin/env bash
GPU_ID=1
data_dir=/home/ubuntu/workspace/datasets/OFFICE31
output_dir=/home/ubuntu/workspace/project/transferlearning/code/DeepDA/output/CFD
# Office31
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config CFD/CFD.yaml --data_dir $data_dir --src_domain dslr --tgt_domain amazon | tee "$output_dir/CFD_D2A.log"
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config CFD/CFD.yaml --data_dir $data_dir --src_domain dslr --tgt_domain webcam | tee "$output_dir/CFD_D2W.log"

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config CFD/CFD.yaml --data_dir $data_dir --src_domain amazon --tgt_domain dslr | tee "$output_dir/CFD_A2D.log"
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config CFD/CFD.yaml --data_dir $data_dir --src_domain amazon --tgt_domain webcam | tee "$output_dir/CFD_A2W.log"

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config CFD/CFD.yaml --data_dir $data_dir --src_domain webcam --tgt_domain amazon | tee "$output_dir/CFD_W2A.log"
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config CFD/CFD.yaml --data_dir $data_dir --src_domain webcam --tgt_domain dslr | tee "$output_dir/CFD_W2D.log"



#data_dir=/home/houwx/tl/datasets/office-home
# Office-Home
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config CFD/CFD.yaml --data_dir $data_dir --src_domain Art --tgt_domain Clipart | tee CFD_A2C.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config CFD/CFD.yaml --data_dir $data_dir --src_domain Art --tgt_domain Real_World | tee CFD_A2R.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config CFD/CFD.yaml --data_dir $data_dir --src_domain Art --tgt_domain Product | tee CFD_A2P.log

#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config CFD/CFD.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain Art | tee CFD_C2A.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config CFD/CFD.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain Real_World | tee CFD_C2R.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config CFD/CFD.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain Product | tee CFD_C2P.log

#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config CFD/CFD.yaml --data_dir $data_dir --src_domain Product --tgt_domain Art | tee CFD_P2A.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config CFD/CFD.yaml --data_dir $data_dir --src_domain Product --tgt_domain Real_World | tee CFD_P2R.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config CFD/CFD.yaml --data_dir $data_dir --src_domain Product --tgt_domain Clipart | tee CFD_P2C.log

#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config CFD/CFD.yaml --data_dir $data_dir --src_domain Real_World --tgt_domain Art | tee CFD_R2A.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config CFD/CFD.yaml --data_dir $data_dir --src_domain Real_World --tgt_domain Product | tee CFD_R2P.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config CFD/CFD.yaml --data_dir $data_dir --src_domain Real_World --tgt_domain Clipart | tee CFD_R2C.log
