
import numpy as np
import os
import copy
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import random
import torch.distributed as dist


def ring_topo_construct():
    comm_list = []
    last_rank = dist.get_world_size()-1
    comm_list.append(dist.new_group([last_rank, 0, 1]))
    for rank in range(dist.get_world_size()-2):
        comm_list.append(dist.new_group([rank, rank+1, rank+2]))
    comm_list.append(dist.new_group([last_rank-1, last_rank, 0]))
    src_rank = dist.get_rank(comm_list[dist.get_rank()])
    src_rank_list = [torch.tensor(i).cuda() for i in range(dist.get_world_size())]
    dist.all_gather(src_rank_list, torch.tensor(src_rank).cuda())
    print('=====', dist.get_rank(), src_rank_list, src_rank_list[dist.get_rank()])
    return comm_list, src_rank, src_rank_list


def bipa_topo_construct():
    world_size = dist.get_world_size()
    even_ranks = [i for i in range(world_size) if i % 2 == 0]
    odd_ranks = [i for i in range(world_size) if i % 2 == 1]
    comm_list = []
    for rank in range(world_size):
        if rank in even_ranks:
            comm_list.append(dist.new_group([rank]+odd_ranks))
        elif rank in odd_ranks:
            comm_list.append(dist.new_group([rank]+even_ranks))
    print('=====', dist.get_rank(), comm_list[dist.get_rank()], dist.get_rank(comm_list[dist.get_rank()]))
    return comm_list


def star_topo_construct():
    world_size = dist.get_world_size()
    comm_list = [None]
    for rank in range(1, world_size):
        comm_list.append(dist.new_group([0,rank]))
    return comm_list


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def average_bn(model, net, group=None):
    group_size = dist.get_world_size(group=group)
    for modellayer, netlayer in zip(model.modules(), net.modules()):
        if isinstance(modellayer, torch.nn.BatchNorm2d):
            running_mean = copy.deepcopy(modellayer.running_mean.data)
            running_var = copy.deepcopy(modellayer.running_var.data)
            dist.all_reduce(running_mean, group=group)
            dist.all_reduce(running_var, group=group)
            netlayer.running_mean.data = running_mean/group_size      
            netlayer.running_var.data = running_var/group_size
