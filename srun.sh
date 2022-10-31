#!/bin/bash

# convex lr
topo='ring'
delay='random_32'
lr_list='0.1 0.2 0.3 0.5'
scheduler='no'
bs='256'
model='linearnet'
dataset='mnist'
seed='42'
epoch='200'
for lr in $lr_list
do
    srun bash run.sh $lr $delay $scheduler $bs $topo $model $dataset $seed $epoch 2>&1 | tee ./log_${model}_${SLURM_NNODES}_node_${topo}_bs${bs}_lr${lr}_sch${scheduler}_${delay}_delay_${model}_model_${dataset}_dataset_${seed}_seed.out
    sleep 5
done
# convex delay
lr='0.1'
delay_list='random_16 random_64 random_128 no'
for delay in $delay_list
do
    srun bash run.sh $lr $delay $scheduler $bs $topo $model $dataset $seed $epoch 2>&1 | tee ./log_${model}_${SLURM_NNODES}_node_${topo}_bs${bs}_lr${lr}_sch${scheduler}_${delay}_delay_${model}_model_${dataset}_dataset_${seed}_seed.out
    sleep 5
done
# convex topo
delay='random_32'
topo_list="star bipartite complete"
for topo in $topo_list
do
    srun bash run.sh $lr $delay $scheduler $bs $topo $model $dataset $seed $epoch 2>&1 | tee ./log_${model}_${SLURM_NNODES}_node_${topo}_bs${bs}_lr${lr}_sch${scheduler}_${delay}_delay_${model}_model_${dataset}_dataset_${seed}_seed.out
    sleep 5
done


# resnet lr
topo='ring'
delay='random_32'
lr_list='0.1 0.2 0.3 0.5'
scheduler='decay_0.01'
bs='256'
model='resnet'
dataset_list='cifar100 cifar10 tiny-imagenet'
seed='42'
epoch='200'
for dataset in $dataset_list
do
    for lr in $lr_list
    do
        srun bash run.sh $lr $delay $scheduler $bs $topo $model $dataset $seed $epoch 2>&1 | tee ./log_${model}_${SLURM_NNODES}_node_${topo}_bs${bs}_lr${lr}_sch${scheduler}_${delay}_delay_${model}_model_${dataset}_dataset_${seed}_seed.out
        sleep 5
    done
done
# resnet delay
lr='0.1'
delay_list='random_16 random_64 random_128 no'
for dataset in $dataset_list
do
    for delay in $delay_list
    do
        srun bash run.sh $lr $delay $scheduler $bs $topo $model $dataset $seed $epoch 2>&1 | tee ./log_${model}_${SLURM_NNODES}_node_${topo}_bs${bs}_lr${lr}_sch${scheduler}_${delay}_delay_${model}_model_${dataset}_dataset_${seed}_seed.out
        sleep 5
    done
done
# resnet topo
delay='random_32'
topo_list="star bipartite complete"
for dataset in $dataset_list
do
    for topo in $topo_list
    do
        srun bash run.sh $lr $delay $scheduler $bs $topo $model $dataset $seed $epoch 2>&1 | tee ./log_${model}_${SLURM_NNODES}_node_${topo}_bs${bs}_lr${lr}_sch${scheduler}_${delay}_delay_${model}_model_${dataset}_dataset_${seed}_seed.out
        sleep 5
    done
done


# vgg lr
topo='ring'
delay='random_32'
lr_list='0.1 0.2 0.3 0.5'
scheduler='no'
bs='256'
model='vgg'
dataset_list='cifar100 cifar10 tiny-imagenet'
seed='42'
epoch='200'
for dataset in $dataset_list
do
    for lr in $lr_list
    do
        srun bash run.sh $lr $delay $scheduler $bs $topo $model $dataset $seed $epoch 2>&1 | tee ./log_${model}_${SLURM_NNODES}_node_${topo}_bs${bs}_lr${lr}_sch${scheduler}_${delay}_delay_${model}_model_${dataset}_dataset_${seed}_seed.out
        sleep 5
    done
done
# vgg delay
lr='0.1'
delay_list='random_16 random_64 random_128 no'
for dataset in $dataset_list
do
    for delay in $delay_list
    do
        srun bash run.sh $lr $delay $scheduler $bs $topo $model $dataset $seed $epoch 2>&1 | tee ./log_${model}_${SLURM_NNODES}_node_${topo}_bs${bs}_lr${lr}_sch${scheduler}_${delay}_delay_${model}_model_${dataset}_dataset_${seed}_seed.out
        sleep 5
    done
done
# vgg topo
delay='random_32'
topo_list="star bipartite complete"
for dataset in $dataset_list
do
    for topo in $topo_list
    do
        srun bash run.sh $lr $delay $scheduler $bs $topo $model $dataset $seed $epoch 2>&1 | tee ./log_${model}_${SLURM_NNODES}_node_${topo}_bs${bs}_lr${lr}_sch${scheduler}_${delay}_delay_${model}_model_${dataset}_dataset_${seed}_seed.out
        sleep 5
    done
done