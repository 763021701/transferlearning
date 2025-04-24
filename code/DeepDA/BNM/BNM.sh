#!/usr/bin/env bash
data_dir=/home/ubuntu/workspace/datasets/OFFICE31
output_dir=/home/ubuntu/workspace/project/transferlearning/code/DeepDA/output/BNM

if [ ! -d "$output_dir" ]; then
    echo "Creating output_dir $output_dir"
    mkdir -p "$output_dir"
fi

# Office31
parallel -j 6 ::: \
"CUDA_VISIBLE_DEVICES=1 python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain dslr --tgt_domain amazon | tee \"$output_dir/BNM_D2A.log\"" \
"CUDA_VISIBLE_DEVICES=2 python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain dslr --tgt_domain webcam | tee \"$output_dir/BNM_D2W.log\"" \
"CUDA_VISIBLE_DEVICES=3 python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain amazon --tgt_domain dslr | tee \"$output_dir/BNM_A2D.log\"" \
"CUDA_VISIBLE_DEVICES=1 python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain amazon --tgt_domain webcam | tee \"$output_dir/BNM_A2W.log\"" \
"CUDA_VISIBLE_DEVICES=2 python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain webcam --tgt_domain amazon | tee \"$output_dir/BNM_W2A.log\"" \
"CUDA_VISIBLE_DEVICES=3 python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain webcam --tgt_domain dslr | tee \"$output_dir/BNM_W2D.log\""


#data_dir=/data/jindwang/OfficeHome
# # Office-Home
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Art --tgt_domain Clipart | tee BNM_A2C.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Art --tgt_domain RealWorld | tee BNM_A2R.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Art --tgt_domain Product | tee BNM_A2P.log

#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain Art | tee BNM_C2A.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain RealWorld | tee BNM_C2R.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain Product | tee BNM_C2P.log

#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Product --tgt_domain Art | tee BNM_P2A.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Product --tgt_domain RealWorld | tee BNM_P2R.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Product --tgt_domain Clipart | tee BNM_P2C.log

#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain RealWorld --tgt_domain Art | tee BNM_R2A.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain RealWorld --tgt_domain Product | tee BNM_R2P.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain RealWorld --tgt_domain Clipart | tee BNM_R2C.log
