dataset='PACS'
algorithm=('MLDG' 'ERM' 'DANN' 'RSC' 'Mixup' 'MMD' 'CORAL' 'VREx' 'CFD')
test_envs=2
gpu_id=0
n_workers=8
data_dir='~/workspace/datasets/Homework3-PACS/PACS/'
max_epoch=120
batch_size=32
net='resnet18'
net_pt_weight='default'
task='img_dg'

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
i=8
cfd_gamma=1
lr=0.005
cfd_alpha=0.8
cfd_beta=0.2
output="/home/ubuntu/workspace/project/transferlearning/code/DeepDG/output/${algorithm[$i]}[$net_pt_weight]-$net-$lr-$cfd_gamma-$cfd_alpha-$cfd_beta-$dataset$test_envs"

python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output \
                --test_envs $test_envs --dataset $dataset --algorithm ${algorithm[i]} --lr $lr \
                --cfd_gamma $cfd_gamma --cfd_alpha $cfd_alpha --cfd_beta $cfd_beta --batch_size $batch_size \
                --N_WORKERS $n_workers --gpu_id $gpu_id --net_pt_weight $net_pt_weight
