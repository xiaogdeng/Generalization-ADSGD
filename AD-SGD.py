
import sys
import numpy as np
import os
import copy
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.distributed as dist
import argparse
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from comms import *
from models import resnet18, vgg16_bn, MNISTNet, LinearNet, vgg16_bn_Ima
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--lr", type=float, default=0.1,
                        help='training learning rate.')
    parser.add_argument("--delay_type", type=str, default='random_32',
                        help='choices: no|random_x. random_x means the maximum asynchronous delay is x. no means no asynchronous.')
    parser.add_argument("--scheduler", type=str, default='no',
                        help='choices: no|decay_x. decay_x means the learning rate decay is 1/(1+x*(iteration+1). no means no lr decay.')
    parser.add_argument("--batchsize", type=int, default=256,
                        help='training and eval batch size.')
    parser.add_argument("--topo", type=str, default='ring', choices=['ring', 'bipartite', 'star', 'complete'],
                        help='communication topology.')
    parser.add_argument("--lr_interval", type=int, default=100,
                        help='learning rate update interval.')
    parser.add_argument("--epoch", type=int, default=200,
                        help='number of train epochs.')
    parser.add_argument("--seed", type=int, default=42,
                        help='random seed.')
    parser.add_argument("--model", type=str, default='vgg', choices=['resnet', 'vgg', 'mnistnet', 'linearnet'],
                        help='model architecture.')
    parser.add_argument("--dataset", type=str, default='cifar10', choices=['cifar10', 'tiny-imagenet', 'cifar100', 'mnist'],
                        help='dataset name and dataset path, The cifar10, cifar100, and mnist datasets can be downloaded automatically by torchvision.datasets. '
                             'tiny-imagenet need downloading maunlly, plz check readme.')
    args = parser.parse_args()
    dist.init_process_group('nccl')
    torch.cuda.set_device(args.local_rank)
    world_size, global_rank = dist.get_world_size(), dist.get_rank()

    writer = SummaryWriter(
        log_dir=f'./tensorboard_runs/{args.model}_runs/{args.topo}_runs/{world_size // 4}node_bs{args.batchsize}_epoch{args.epoch}_lr{args.lr}_sch{args.scheduler}_{args.delay_type}_delay_{args.model}_{args.dataset}')

    print(
        f'==> lr: {args.lr}--asyn: {args.delay_type}--schedul: {args.scheduler}--bs: {args.batchsize}--topo: {args.topo}--epoch: {args.epoch}--seed: {args.seed}--model: {args.model}--data: {args.dataset}')
    print(f'==> Using {args.topo} topo...')
    if args.topo == 'ring':
        if world_size < 3:
            comm_list = [None]
        else:
            comm_list, src_rank, src_rank_list = ring_topo_construct()
    elif args.topo == 'bipartite':
        comm_list = bipa_topo_construct()
    elif args.topo == 'star':
        comm_list = star_topo_construct()

    seed_torch(args.seed + global_rank)
    global best_acc
    best_acc = 0.
    start_epoch = 0

    # Load data
    print('==> Preparing data...')
    root = "Your dataset root"

    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = torchvision.datasets.MNIST(root=root, train=True,
                                               download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=root, train=False,
                                              download=True, transform=transform)
    elif args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_set = torchvision.datasets.CIFAR10(root=root, train=True,
                                                 download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root=root, train=False,
                                                download=True, transform=transform_test)
    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                 std=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343])
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                 std=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343])
        ])
        train_set = torchvision.datasets.CIFAR100(root=root, train=True,
                                                  download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR100(root=root, train=False,
                                                 download=True, transform=transform_test)
    elif args.dataset == 'tiny-imagenet':
        transform_train = transforms.Compose([
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        train_set = torchvision.datasets.ImageFolder(root=os.path.join(root, 'tiny-imagenet-200/train'),
                                                     transform=transform_train)
        test_set = torchvision.datasets.ImageFolder(root=os.path.join(root, 'tiny-imagenet-200/val'),
                                                    transform=transform_test)
    else:
        raise ValueError(args.dataset + ' is not known.')

    kwargs = {'num_workers': 1, 'pin_memory': True}
    batchsize_test = args.batchsize
    print('Batch size of the test set: ', batchsize_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batchsize_test,
                                              shuffle=False, **kwargs)
    batchsize_train = args.batchsize
    print('Batch size of the train set: ', batchsize_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batchsize_train,
                                               shuffle=True, **kwargs)

    if args.model == 'resnet' and args.dataset == 'cifar10':
        net = resnet18(10).cuda()
    elif args.model == 'resnet' and args.dataset == 'cifar100':
        net = resnet18(100).cuda()
    elif args.model == 'vgg' and args.dataset == 'cifar10':
        net = vgg16_bn(10).cuda()
    elif args.model == 'vgg' and args.dataset == 'cifar100':
        net = vgg16_bn(100).cuda()
    elif args.model == 'vgg' and args.dataset == 'tiny-imagenet':
        net = vgg16_bn_Ima(200).cuda()
    elif args.model == 'resnet' and args.dataset == 'tiny-imagenet':
        net = resnet18(200).cuda()
    elif args.model == 'mnistnet':
        net = MNISTNet().cuda()
    elif args.model == 'linearnet':
        net = LinearNet().cuda()
    else:
        raise ValueError(args.model + ' is not known.')

    for p in net.parameters():
        dist.all_reduce(p.data)
        p.data /= world_size
    average_bn(net, net)
    model = copy.deepcopy(net)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    if args.scheduler == 'no':
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda iteration: 1)
    elif args.scheduler.startswith('decay'):
        lr_decay = float(args.scheduler.split('_')[-1])
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda iteration: 1 / (1 + lr_decay * (iteration + 1)))
    else:
        raise ValueError(args.scheduler + ' is not known.')

    step = 0
    nepoch = args.epoch
    global_step = 0
    update_count = 0
    lr_interval = args.lr_interval
    writer.add_scalar(f'Lr/{global_rank}', lr_scheduler.get_lr()[0], global_step)

    if args.delay_type == 'no':
        delay_interval = 1
    elif args.delay_type.startswith('random'):
        max_interval = int(args.delay_type.split('_')[1])
        if global_rank == world_size - 1:
            delay_interval = 1
        else:
            delay_interval = random.randint(1, max_interval)
        print('rank & asynchronous delay is', global_rank, delay_interval)
    else:
        raise ValueError(args.delay_type + ' is not known.')

    for epoch in range(nepoch):
        model.train()
        correct = 0
        total = 0
        train_loss = 0.
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            x, target = Variable(x.cuda()), Variable(target.cuda())
            score, loss = model(x, target)
            loss.backward()
            train_loss += loss.item()
            _, predicted = torch.max(score.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
            for local_step in range(delay_interval):
                # model average
                model_state = [para.data for para in model.parameters()]
                coalesced = _flatten_dense_tensors(model_state)
                new_coalesced = copy.deepcopy(coalesced)
                if local_step == delay_interval - 1:
                    update = torch.Tensor([1]).int().cuda()
                else:
                    update = torch.Tensor([0]).int().cuda()
                update_status = [torch.zeros(1, dtype=torch.int32).cuda() for _ in range(world_size)]
                dist.all_gather(update_status, update)
                if args.topo == 'ring' or args.topo == 'bipartite':
                    for idx in range(world_size):
                        if update_status[idx] == 1:
                            dist.all_reduce(new_coalesced, group=comm_list[idx])
                            if dist.get_rank(comm_list[idx]) > -1:
                                group_size = dist.get_world_size(group=comm_list[idx])
                                new_coalesced /= group_size
                                for buf, synced in zip(model_state,
                                                       _unflatten_dense_tensors(new_coalesced, model_state)):
                                    buf.copy_(synced)
                                model_state = [para.data for para in model.parameters()]
                                coalesced = _flatten_dense_tensors(model_state)
                                new_coalesced = copy.deepcopy(coalesced)
                elif args.topo == 'star':
                    if update_status[0] == 1:
                        dist.all_reduce(new_coalesced)
                        new_coalesced /= world_size
                        for buf, synced in zip(model_state, _unflatten_dense_tensors(new_coalesced, model_state)):
                            buf.copy_(synced)
                        model_state = [para.data for para in model.parameters()]
                        coalesced = _flatten_dense_tensors(model_state)
                        new_coalesced = copy.deepcopy(coalesced)
                    for idx in range(1, world_size):
                        if update_status[idx] == 1:
                            if dist.get_rank(comm_list[idx]) > -1:
                                if global_rank != 0:
                                    new_coalesced *= (world_size - 1)
                                dist.all_reduce(new_coalesced, group=comm_list[idx])
                                new_coalesced /= world_size
                                for buf, synced in zip(model_state,
                                                       _unflatten_dense_tensors(new_coalesced, model_state)):
                                    buf.copy_(synced)
                                model_state = [para.data for para in model.parameters()]
                                coalesced = _flatten_dense_tensors(model_state)
                                new_coalesced = copy.deepcopy(coalesced)
                elif args.topo == 'complete':
                    dist.all_reduce(new_coalesced)
                    new_coalesced /= world_size
                    for buf, synced in zip(model_state, _unflatten_dense_tensors(new_coalesced, model_state)):
                        buf.copy_(synced)

                global_step += 1
                if global_step % 100 == 0:
                    avg_train_loss = train_loss / total
                    print('Worker: %02d | Epoch: %03d | Global Step: %05d | Training Loss: %.4f'
                          % (global_rank, epoch, global_step, avg_train_loss * args.batchsize))

                if local_step == delay_interval - 1:
                    optimizer.step()
                dist.all_reduce(update)
                update_count += int(update.cpu())
                if update_count % lr_interval == 0:
                    lr_scheduler.step()
                    writer.add_scalar(f'Lr/{global_rank}', lr_scheduler.get_lr()[0], global_step)

                if global_step % 1000 == 0:
                    # get global model
                    global_net_state = [para.data for para in net.parameters()]
                    eval_model_state = [para.data for para in model.parameters()]
                    eval_coalesced = copy.deepcopy(_flatten_dense_tensors(eval_model_state))
                    dist.reduce(eval_coalesced, dst=0)
                    eval_coalesced /= world_size
                    for buf, synced in zip(global_net_state,
                                           _unflatten_dense_tensors(eval_coalesced, global_net_state)):
                        buf.copy_(synced)

                    average_bn(model, net)
                    # net eval, use the global model
                    model.eval()
                    if global_rank == 0:
                        net.eval()
                        global_train_loss = 0.
                        global_train_correct = 0
                        global_train_total = 0
                        for _, (x, target) in enumerate(train_loader):
                            x, target = Variable(x.cuda()), Variable(target.cuda())
                            score, loss = net(x, target)
                            global_train_loss += loss.item()
                            _, predicted = torch.max(score.data, 1)
                            global_train_total += target.size(0)
                            global_train_correct += predicted.eq(target.data).cpu().sum()

                        global_avg_train_loss = global_train_loss / global_train_total
                        acc = (1. * global_train_correct.data.cpu().numpy()) / global_train_total
                        train_acc = acc
                        print('Worker: %02d | Global Train Epoch: %03d | Global Step: %05d | Training Loss: %.4f | Training accuracy: %.4f'
                              % (global_rank, epoch, global_step, global_avg_train_loss * args.batchsize, acc))
                        writer.add_scalar('Accuracy/train', acc, global_step)
                        writer.add_scalar('Loss/train', global_avg_train_loss * args.batchsize, global_step)

                        test_loss = 0.
                        test_correct = 0
                        test_total = 0
                        for eval_batch_idx, (x, target) in enumerate(test_loader):
                            x, target = x.cuda(), target.cuda()
                            score, loss = net(x, target)
                            test_loss += loss.item()
                            _, predicted = torch.max(score.data, 1)
                            test_total += target.size(0)
                            test_correct += predicted.eq(target.data).cpu().sum()

                        avg_test_loss = test_loss / test_total
                        acc = (1. * test_correct.data.cpu().numpy()) / 10000.
                        print('Worker: %02d | Global Evaluation Epoch: %03d | Global Step: %05d | Testing Loss: %.4f | Testing accuracy: %.4f'
                              % (global_rank, epoch, global_step, avg_test_loss * args.batchsize, acc))
                        writer.add_scalar('Accuracy/test', acc, global_step)
                        writer.add_scalar('Loss/test', avg_test_loss * args.batchsize, global_step)
                        writer.add_scalar('Generalization_error',
                                          np.abs(avg_test_loss - global_avg_train_loss) * args.batchsize, global_step)

                        if acc > best_acc:
                            best_acc = acc
                        print('The best acc: ', best_acc)

                    dist.barrier()
                    # model eval
                    model.eval()
                    test_loss = 0.
                    test_correct = 0
                    test_total = 0
                    for eval_batch_idx, (x, target) in enumerate(test_loader):
                        x, target = x.cuda(), target.cuda()
                        score, loss = model(x, target)
                        test_loss += loss.item()
                        _, predicted = torch.max(score.data, 1)
                        test_total += target.size(0)
                        test_correct += predicted.eq(target.data).cpu().sum()
                    avg_test_loss = test_loss / test_total
                    acc = (1. * test_correct.data.cpu().numpy()) / 10000.
                    print('Worker: %02d | Local Evaluation Epoch: %03d | Global Step: %05d | Testing Loss: %.4f | Testing accuracy: %.4f'
                          % (global_rank, epoch, global_step, avg_test_loss * args.batchsize, acc))

                    model.train()
                if epoch == nepoch - 1 and batch_idx == len(train_loader) - 1:
                    exit_sig = torch.Tensor([1]).int().cuda()
                else:
                    exit_sig = torch.Tensor([0]).int().cuda()
                dist.all_reduce(exit_sig)
                if int(exit_sig.cpu()) >= 1:
                    sys.exit()
