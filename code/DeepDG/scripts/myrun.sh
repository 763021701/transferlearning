dataset='PACS'
algorithm=('MLDG' 'ERM' 'DANN' 'RSC' 'Mixup' 'MMD' 'CORAL' 'VREx' 'CFD')
test_envs=1
gpu_ids=0
data_dir='~/workspace/datasets/Homework3-PACS/PACS/'
max_epoch=120
net='resnet18'
task='img_dg'
output='/home/ubuntu/workspace/project/transferlearning/code/DeepDG/output/CFD-gamma0.5'

# MLDG
#i=0
#python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output \
#--test_envs $test_envs --dataset $dataset --algorithm ${algorithm[i]} --mldg_beta 10

# DANN
#python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output \
#--test_envs $test_envs --dataset $dataset --algorithm ${algorithm[2]} --lr 0.001 --alpha 1 --N_WORKERS 8

# MMD
#python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output \
#--test_envs $test_envs --dataset $dataset --algorithm ${algorithm[5]} --lr 0.001 --mmd_gamma 1 --N_WORKERS 8

# CFD
python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output \
--test_envs $test_envs --dataset $dataset --algorithm ${algorithm[8]} --lr 0.001 --cfd_gamma 0.5 --N_WORKERS 8
