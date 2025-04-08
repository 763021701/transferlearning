dataset='PACS'
algorithm=('MLDG' 'ERM' 'DANN' 'RSC' 'Mixup' 'MMD' 'CORAL' 'VREx' 'CFD')
algorithm2=('MLDG2' 'ERM2' 'DANN2' 'RSC2' 'Mixup2' 'MMD2' 'CORAL2' 'VREx2' 'CFD2')
test_envs=1
gpu_id=3
n_workers=8
data_dir='~/workspace/datasets/Homework3-PACS/PACS/'
max_epoch=120
batch_size=32
net='vgg16'
net_pt_weight='default'
task='img_dg'

# MMD
#i=5
#mmd_gamma=0.5
#lr=0.005
#output="/home/ubuntu/workspace/project/transferlearning/code/DeepDG/output/${algorithm[$i]}[$net_pt_weight]-$net-$lr-$mmd_gamma-$dataset$test_envs"

#python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output \
#       --test_envs $test_envs --dataset $dataset --algorithm ${algorithm[i]} --lr $lr --mmd_gamma $mmd_gamma \
#       --batch_size $batch_size --N_WORKERS $n_workers --gpu_id $gpu_id --net_pt_weight $net_pt_weight

# MMD2
i=5
mmd_gamma=1
lr=0.005
output="/home/ubuntu/workspace/project/transferlearning/code/DeepDG/output/${algorithm2[$i]}_$net_pt_weight-$net-$lr-$mmd_gamma-$dataset$test_envs"

python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output \
       --test_envs $test_envs --dataset $dataset --algorithm ${algorithm2[i]} --lr $lr --mmd_gamma $mmd_gamma \
       --batch_size $batch_size --N_WORKERS $n_workers --gpu_id $gpu_id --net_pt_weight $net_pt_weight
