dataset='PACS'
algorithm=('MLDG' 'ERM' 'DANN' 'RSC' 'Mixup' 'MMD' 'CORAL' 'VREx' 'CFD')
algorithm2=('MLDG2' 'ERM2' 'DANN2' 'RSC2' 'Mixup2' 'MMD2' 'CORAL2' 'VREx2' 'CFD2')
test_envs=0
gpu_id=3
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

# CFD2
i=8
cfd_gamma=1
lr=0.005
cfd_alpha=0.8
cfd_beta=0.2
bn_only="False"
freeze_bn="False"

if [ "$bn_only" = "True" ]; then
  bn_only_flag="--bn_only"
else
  bn_only_flag=""
fi
if [ "$freeze_bn" = "True" ]; then
  freeze_bn_flag="--freeze_bn"
else
  freeze_bn_flag=""
fi

output="/home/ubuntu/workspace/project/transferlearning/code/DeepDG/output/${algorithm2[$i]}_$net_pt_weight-$bn_only-$freeze_bn-$net-$lr-$cfd_gamma-$cfd_alpha-$cfd_beta-$dataset$test_envs"

python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output \
                --test_envs $test_envs --dataset $dataset --algorithm ${algorithm2[i]} --lr $lr \
                --cfd_gamma $cfd_gamma --cfd_alpha $cfd_alpha --cfd_beta $cfd_beta --batch_size $batch_size \
                --N_WORKERS $n_workers --gpu_id $gpu_id --net_pt_weight $net_pt_weight \
                $bn_only_flag $freeze_bn_flag
