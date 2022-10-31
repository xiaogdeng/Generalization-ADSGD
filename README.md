## Stability-Based Generalization Analysis of the Asynchronous Decentralized SGD

### Environment

```
# GPU environment required

torch>=1.10.0
torchvision>=0.11.1
numpy>=1.19.5
```


### Dataset

The Tiny-ImageNet dataset can be downloaded from [here](https://paperswithcode.com/dataset/tiny-imagenet). 

The CIFAR-10, CIFAR-100, and MNIST datasets can be downloaded automatically by torchvision.datasets.

### Example Usage

```
python -m torch.distributed.launch --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES \
                                   --node_rank $NODE_RANK --master_addr $MASTER_ADDR \
                                   --master_port $MASTER_PORT \
                                   AD-SGD.py --lr=$1 --delay_type=$2 --scheduler=$3 \
                                   --batchsize=$4 --topo=$5 --model=$6 --dataset=$7 \
                                   --seed=$8 --epoch=$9
```


### Usage

```
usage: AD-SGD.py [-h] [--local_rank LOCAL_RANK] [--lr LR]
                 [--delay_type DELAY_TYPE] [--scheduler SCHEDULER]
                 [--batchsize BATCHSIZE]
                 [--topo {ring,bipartite,star,complete}]
                 [--lr_interval LR_INTERVAL] [--epoch EPOCH] [--seed SEED]
                 [--model {resnet,vgg,mnistnet,linearnet}]
                 [--dataset {cifar10,tiny-imagenet,cifar100,mnist}]

optional arguments:
  -h, --help            show this help message and exit
  --local_rank LOCAL_RANK
  --lr LR               training learning rate.
  --delay_type DELAY_TYPE
                        choices: no|random_x. random_x means the maximum
                        asynchronous delay is x. no means no asynchronous.
  --scheduler SCHEDULER
                        choices: no|decay_x. decay_x means the learning rate
                        decay is 1/(1+x*(iteration+1). no means no lr decay.
  --batchsize BATCHSIZE
                        training and eval batch size.
  --topo {ring,bipartite,star,complete}
                        communication topology.
  --lr_interval LR_INTERVAL
                        learning rate update interval.
  --epoch EPOCH         number of train epochs.
  --seed SEED           random seed.
  --model {resnet,vgg,mnistnet,linearnet}
                        model architecture.
  --dataset {cifar10,tiny-imagenet,cifar100,mnist}
                        dataset name and dataset path, The cifar10, cifar100,
                        and mnist datasets can be downloaded automatically by
                        torchvision.datasets. tiny-imagenet need downloading
                        maunlly, plz check readme.
```

