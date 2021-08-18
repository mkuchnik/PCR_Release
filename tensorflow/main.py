import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# New
import pathlib
import loader_util
import main_helper
import pandas as pd
import math
import tensorflow as tf

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
loader_names = sorted(["tensorflow", "pytorch"])

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--data_format',
                    default=loader_util.DEFAULT_DATA_FORMAT,
                    choices=loader_util.FILE_DATA_FORMATS)
parser.add_argument('--scan', default=0, type=int,
                    help='The number of scans to use. 0 is baseline.'
                         '(default: 0)')
parser.add_argument('--train_loader', default="tensorflow", type=str,
                    choices=loader_names,
                    help="The training loader to use")
parser.add_argument('--validation_loader', default="pytorch", type=str,
                    choices=loader_names,
                    help="The training loader to use")
parser.add_argument('--fp16', dest='fp16', action='store_true',
                    help='use reduced-precision (mixed) training')
parser.add_argument('--synthetic_training', dest='synthetic_training',
                    action='store_true',
                    help='re-use fake data')
parser.add_argument('--synthetic_loader', dest='synthetic_loader',
                    action='store_true',
                    help='use fake data')
parser.add_argument('--synthetic_model', dest='synthetic_model',
                    action='store_true',
                    help='use fake model')
parser.add_argument('--validate-freq', default=1, type=int,
                    help='validate frequency (default: 1)')
parser.add_argument('--checkpoint-freq', default=1, type=int,
                    help='checkpoint frequency (default: 1)')
parser.add_argument('--checkpoint_prefix', default='checkpoints/', type=str,
                    help='path to checkpoint prefix (default: none)')
parser.add_argument('--val_data_dir', default=None, type=str,
                    help='manual validation path')
parser.add_argument('--val_data_format',
                    default="files",
                    choices=loader_util.FILE_DATA_FORMATS)
parser.add_argument('--channels_last', dest='channels_last', action='store_true',
                    help='use channels last training')
parser.add_argument('--image_size', default=224, type=int,
                    help='The image size for training')
parser.add_argument('--imagenet_training', dest='imagenet_training', action='store_true',
                    help='whether to use ImageNet specific training (for loader)')
parser.add_argument('--autotune-freq', default=1, type=int,
                    help='autotune frequency (default: 1)')
parser.add_argument('--scale_lr', action='store_true',
                    help='rescale lr using effective batch size')
parser.add_argument('--deterministic', action='store_true',
                    help='Turns on deterministic training (slower)')
parser.add_argument('--sync-freq', default=0, type=int,
                    metavar='N',
                    help='training synchronization frequency (default: 0)')
parser.add_argument('--distributed_validate', default=None, type=bool,
                    help='validate by sharding dataset')
parser.add_argument('--training_preprocessing',
                    default="imagenet",
                    choices=loader_util.PREPROCESSING_CHOICES)

best_acc1 = 0


@tf.autograph.experimental.do_not_convert
def main():
    args = parser.parse_args()

    print(args)

    if args.seed is not None:
        # TODO(mkuchnik): Add GPU rank
        random.seed(args.seed + args.local_rank)
        torch.manual_seed(args.seed + args.local_rank)
        tf.random.set_seed(args.seed + args.local_rank)
        if args.deterministic:
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.distributed and args.distributed_validate is None:
        args.distributed_validate = True

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        print("Using multiprocessing with {} processes".format(ngpus_per_node))
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


@tf.autograph.experimental.do_not_convert
def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.local_rank == -1:
            args.local_rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.local_rank = args.local_rank * ngpus_per_node + gpu
        print("CREATING DISTRIBUTED PROCESS", args.local_rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.local_rank)
        print("CREATED DISTRIBUTED PROCESS", args.local_rank)
    elif args.local_rank < 0:
        print("Overriding local_rank to be 0")
        args.local_rank = 0
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.channels_last:
        print("Using channels last")
        model = model.to(memory_format=torch.channels_last)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            # Rescale tf.data loaders, too
            loader_parallelism = loader_util.get_default_parallelism()
            loader_parallelism = int((loader_parallelism + ngpus_per_node - 1) / ngpus_per_node)
            loader_util.set_default_parallelism(loader_parallelism)
            loader_buffer = loader_util.get_default_buffer_size()
            loader_buffer = int((loader_buffer + ngpus_per_node - 1) / ngpus_per_node)
            loader_util.set_default_buffer_size(loader_buffer)
            print("Setting loader parallelism to {}, buffer_size to "
                  "{} for GPU {}".format(loader_parallelism, loader_buffer,
                                         args.gpu))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()


    print("WORLD_SIZE: {}".format(args.world_size))
    if args.scale_lr and args.world_size > 0:
        # NOTE(mkuchnik): world size includes GPUs and nodes
        effective_batch_size = args.batch_size * args.world_size
        native_batch = 256
        lr_multiple = effective_batch_size / native_batch
        print("Effective batch size: {} ({}x)".format(effective_batch_size,
                                                      lr_multiple))
        scale_lr = args.lr * lr_multiple
        print("Setting learning rate to: {}".format(scale_lr))
        args.lr = scale_lr

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    print("Setting train loader to {}".format(args.train_loader), flush=True)
    train_loader, train_sampler = main_helper.get_train_loader(args) # Train sampler can be None

    print("Using {} validation loader".format(args.validation_loader))
    val_loader = main_helper.get_validation_loader(args)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    column_names = ["time", "epoch", "step", "type", "value"]
    stats_df = pd.DataFrame([], columns=column_names)

    _train_fn = train if not args.synthetic_training else synthetic_train
    def append_stats_df(time_s, epoch, step, type, value):
        new_data = [time_s, epoch, step, type, value]
        stats_df.loc[len(stats_df)] = new_data
    def train_fn(train_loader, model, criterion, optimizer, epoch, args):
        rets = _train_fn(train_loader, model, criterion, optimizer, epoch, args)
        time_s = time.time()
        append_stats_df(time_s, epoch, 0, "train_top_1", float(rets[0]))
        append_stats_df(time_s, epoch, 0, "train_top_5", float(rets[1]))
        append_stats_df(time_s, epoch, 0, "train_loss", float(rets[2]))
    def dump_df_csv(filename):
        stats_df.to_csv(filename, index=False)
    def validate_fn(val_loader, model, criterion, epoch, args):
        acc1, acc5 = validate(val_loader, model, criterion, args)
        time_s = time.time()
        append_stats_df(time_s, epoch, 0, "validate_top_1", float(acc1))
        append_stats_df(time_s, epoch, 0, "validate_top_5", float(acc5))
        return acc1
    def autotune_fn(model, criterion, optimizer, epoch, args, metadata):
        epochs_since_autotune = metadata["epochs_since_autotune"]
        if args.autotune_freq and args.local_rank == 0:
            print("Epochs since autotune: {}".format(epochs_since_autotune))
        autotune_start_epoch = 5
        if ((args.autotune_freq and epoch == autotune_start_epoch)
                or (args.autotune_freq
                    and epochs_since_autotune >= args.autotune_freq
                    and epoch >= autotune_start_epoch)):
            distributed_eval = bool(args.distributed)
            num_batches = 10
            if distributed_eval:
                num_batches = max(math.ceil(num_batches / args.world_size), 1)
            scores = main_helper.calculate_scan_gradient_similarity(
                model,
                criterion,
                optimizer,
                args,
                num_batches=num_batches,
                distributed_eval=distributed_eval)
            if distributed_eval:
                scores_batch = (scores * num_batches).tolist()
                scores_batch.append(num_batches)
                scores_tensor = torch.FloatTensor(scores_batch).cuda(
                    non_blocking=True)
                torch.distributed.all_reduce(scores_tensor,
                                             op=dist.ReduceOp.SUM,
                                             async_op=False)
                scores = scores_tensor.cpu().numpy()
                scores /= scores[-1]
                scores = scores[:-1]

            # TODO(mkuchnik): Take threshold as arg
            filtered_scores = scores > 0.8
            chosen_score = None
            for i, x in enumerate(filtered_scores):
                if x:
                    chosen_score = i
                    break
            if chosen_score is None:
                raise ValueError("Expected a score to be chosen")
            chosen_scan = i + 1
            print("autotune scores: {}, scan {} chosen.".format(scores,
                                                                chosen_scan))
            time_s = time.time()
            metadata["epochs_since_autotune"] = 0
            append_stats_df(time_s, epoch, 0, "autotune_scores", str(scores.tolist()))
            append_stats_df(time_s, epoch, 0, "autotune_scan", chosen_scan)
        else:
            metadata["epochs_since_autotune"] += 1
            time_s = time.time()
            append_stats_df(time_s, epoch, 0, "autotune_scan", args.scan)
            chosen_scan = args.scan
        return metadata, chosen_scan


    if args.checkpoint_prefix:
        checkpoint_prefix = pathlib.Path(args.checkpoint_prefix).as_posix()
        pathlib.Path(checkpoint_prefix).mkdir(parents=True, exist_ok=True)
        checkpoint_prefix += "/"
    else:
        checkpoint_prefix = ""

    # Validate before training to get initial point
    acc1 = validate_fn(val_loader, model, criterion, 0, args)
    print("Starting training".format(args.local_rank))
    swap_loaders = True
    autotune_metadata = {"epochs_since_autotune": 0}
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and train_sampler:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # Possible perform autotuning
        autotune_metadata, chosen_scan = autotune_fn(model, criterion, optimizer, epoch, args, autotune_metadata)

        # Swap loader from autotune selection
        if chosen_scan != args.scan and swap_loaders:
            args.scan = chosen_scan
            del train_loader
            train_loader, train_sampler = main_helper.get_train_loader(args) # Train sampler can be None

        # train for one epoch
        train_fn(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        if args.validate_freq and epoch % args.validate_freq == 0:
            acc1 = validate_fn(val_loader, model, criterion, epoch, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
        else:
            is_best = False

        if ((not args.multiprocessing_distributed or
            (args.multiprocessing_distributed
                and args.local_rank % ngpus_per_node == 0))
            and args.checkpoint_freq and epoch % args.checkpoint_freq == 0):
            save_checkpoint_simple({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, filename="{}checkpoint_{}_{}.pth.tar".format(
                checkpoint_prefix, epoch+1, args.local_rank))
            dump_df_csv("{}train_stats_{}.csv".format(checkpoint_prefix,
                                                      args.local_rank))

def synthetic_train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    len_train_loader = len(train_loader)
    progress = ProgressMeter(
        len_train_loader,
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()

    # switch to train mode
    model.train()

    total_start_time = time.time()
    end = time.time()
    images = None
    target = None
    for i, (images, target) in enumerate(train_loader):
        break
    for i in range(len(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        if args.channels_last:
            images = images.to(memory_format=torch.channels_last)

        # compute output
        if not args.synthetic_model:
            output = model(images)
            if args.fp16:
                with torch.cuda.amp.autocast():
                    loss = criterion(output, target)
            else:
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            if args.fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        if args.sync_freq and i % args.sync_freq == 0:
            torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    total_load_time = data_time.sum
    percentage_load = total_load_time / total_time
    train_msg = (
        "({rank}) Load: {total_load_time}s, "
        "Batch: {total_batch_time}s, "
        "Total: {total_time} s ({steps_s} steps/s, {percentage_load:%} loader)"
        .format(rank=args.local_rank, total_load_time=total_load_time,
                total_batch_time=total_batch_time, total_time=total_time,
                steps_s=len(train_loader) / total_time,
                percentage_load=percentage_load))
    print(train_msg)

    return top1.avg, top5.avg, losses.avg

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    len_train_loader = len(train_loader)
    progress = ProgressMeter(
        len_train_loader,
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}] ({})".format(epoch, args.local_rank))

    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()

    # switch to train mode
    model.train()

    total_start_time = time.time()
    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        if args.channels_last:
            images = images.to(memory_format=torch.channels_last)

        # compute output
        if not args.synthetic_model:
            output = model(images)
            if args.fp16:
                with torch.cuda.amp.autocast():
                    loss = criterion(output, target)
            else:
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            if args.fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        if args.sync_freq and i % args.sync_freq == 0:
            torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.local_rank == 0:
            progress.display(i)
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    total_load_time = data_time.sum
    total_batch_time = batch_time.sum
    percentage_load = total_load_time / total_time

    train_msg = (
        "({rank}) Load: {total_load_time}s, "
        "Batch: {total_batch_time}s, "
        "Total: {total_time} s ({steps_s} steps/s, {percentage_load:%} loader)"
        .format(rank=args.local_rank, total_load_time=total_load_time,
                total_batch_time=total_batch_time, total_time=total_time,
                steps_s=len(train_loader) / total_time,
                percentage_load=percentage_load))
    print(train_msg)

    return top1.avg, top5.avg, losses.avg

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ({})'.format(args.local_rank))

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        if args.distributed_validate:
            # Acc1, Acc5, Batch size
            def create_res_tensor():
                res_tensor = torch.zeros(3, dtype=torch.int32).cuda(
                    non_blocking=True)
                return res_tensor
            def sync_res_tensor(res_tensor):
                res_tensor = res_tensor.cuda(non_blocking=True)
                torch.distributed.all_reduce(res_tensor, op=dist.ReduceOp.SUM,
                                             async_op=False)
                return res_tensor
            def delayed_top_update(res_tensor):
                res_tensor = sync_res_tensor(res_tensor)
                res_tensor = res_tensor.cpu()
                # print("Res Tensor: {} ({})".format(res_tensor, args.local_rank))
                acc1 = res_tensor[0]
                acc5 = res_tensor[1]
                aggregate_batch_size = res_tensor[2]
                res_tensor = create_res_tensor()
                top1.update(acc1 * 100.0 / aggregate_batch_size,
                            aggregate_batch_size)
                top5.update(acc5 * 100.0 / aggregate_batch_size,
                            aggregate_batch_size)
                return res_tensor
            res_tensor = create_res_tensor()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            if args.channels_last:
                images = images.to(memory_format=torch.channels_last)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            if not args.distributed_validate:
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
            else:
                # measure accuracy and record loss in distributed fashion
                # Delay updates until print or finalize
                acc1, acc5, aggregate_batch_size = \
                    distributed_accuracy_helper(output, target, topk=(1, 5))
                res_tensor = res_tensor.cuda(non_blocking=True)
                res_tensor[0] += acc1[0]
                res_tensor[1] += acc5[0]
                res_tensor[2] += aggregate_batch_size
            losses.update(loss.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                # NOTE(mkuchnik): We disable sync because it is hard not to
                # deadlock if number of shards differs between nodes
                #if args.distributed_validate:
                #    res_tensor = delayed_top_update(res_tensor)
                progress.display(i)

        if args.distributed_validate:
            res_tensor = delayed_top_update(res_tensor)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} ({rank})'
              .format(top1=top1, top5=top5, rank=args.local_rank))

    return top1.avg, top5.avg

def save_checkpoint_simple(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        if self.count:
            return self.sum / self.count
        else:
            return 0

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(avg=self.avg, **self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def distributed_accuracy_helper(output, target, topk=(1,), rank=None):
    """Computes the accuracy over the k top predictions for the specified values of k
    For distributed implementation, do the following:

    res_tensor = torch.tensor(res).cuda()
    # Now should be sum of all ranks
    if rank is None:
        torch.distributed.all_reduce(res_tensor, async_op=True)
    else:
        torch.distributed.reduce(res_tensor, rank)
    # NOTE(mkuchnik): Last dimension (batch) is still there
    res = res_tensor.tolist()


    WARNING: This code will deadlock if a validation dataset is short on one
    rank. It is therefore better to aggregate into a tensor before doing a
    single communication.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).sum(0, keepdim=True)
            res.append(correct_k)

        res.append(batch_size) # Batch is last dim
        return res


if __name__ == '__main__':
    main()
