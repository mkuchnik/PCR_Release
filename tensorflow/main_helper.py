"""
Helper functions to keep main smaller
"""

import os
import pathlib
import math
import re

import tensorflow as tf
import torch
import numpy as np
import data_prefetcher
import torchvision.datasets as datasets
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import loader_util

def cosine_similarity_f(x1, x2):
    if isinstance(x1, np.ndarray):
        dot = np.dot(x1, x2)
        norm = np.linalg.norm(x1) * np.linalg.norm(x2)
        cos_sim = dot / norm
    else:
        dist_f = torch.nn.CosineSimilarity(dim=0)
        cos_sim = dist_f(x1, x2)
    return  cos_sim

def get_model_gradient_vector(model):
    ww = []
    for name, W in model.named_parameters():
        ww.append(torch.flatten(W.grad))
    return torch.cat(ww).detach()

def calculate_model_gradient_vector(train_loader, model, criterion, optimizer,
                                    args, num_batches):
    """Trains model for a few steps and accumulates the gradient"""
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()

    # switch to train mode
    model.train()

    optimizer.zero_grad()
    for i, (images, target) in enumerate(train_loader):
        if i >= num_batches:
            break
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)
        if args.channels_last:
            images = images.to(memory_format=torch.channels_last)

        # compute output
        output = model(images)
        if args.fp16:
            with torch.cuda.amp.autocast():
                loss = criterion(output, target)
        else:
            loss = criterion(output, target)

        # compute gradient and do SGD step
        if args.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    model_grad = get_model_gradient_vector(model)
    return model_grad


def calculate_scan_gradient_similarity(model, criterion, optimizer,
                                       args, max_scans: int=10,
                                       num_batches=10,
                                       distributed_eval=False):
    """
    Calculates the gradient cosine similarity for various scans
    and returns numpy array of the scores
    """
    worker_index = None
    world_size = None
    if distributed_eval:
        worker_index = args.local_rank
        world_size = args.world_size
    if ((worker_index is not None or world_size is not None)
            and (not worker_index is not None or not world_size is not None)):
        raise ValueError("worker_index and world_size must both be set: {},"
                         " {}".format(worker_index, world_size))
    dataset_path = args.data
    root_path = pathlib.Path(dataset_path).parent
    if args.data_format == "PCR":
        index_filename = (root_path / "PCR_index.pb").as_posix()
        ds = loader_util.get_raw_tfdata_loader(dataset_path=args.data,
                                               data_format=args.data_format,
                                               scans=10,
                                               deterministic=True,
                                               index_filename=index_filename,
                                               worker_index=worker_index,
                                               world_size=world_size)
    else:
        ds = loader_util.get_raw_tfdata_loader(dataset_path=args.data,
                                               data_format=args.data_format,
                                               deterministic=True,
                                               worker_index=worker_index,
                                               world_size=world_size)
    raw_data = [x for x in ds.batch(args.batch_size * num_batches).take(1)][0]
    del ds
    scan_view_dataset = loader_util.ScanViewDataset(raw_data)
    base_dataset = \
        scan_view_dataset.get_baseline_dataset(image_size=args.image_size,
                                               batch_size=args.batch_size,
                                               permute=True,
                                               image_normalization=True,
                                               randomized_image_postprocessing=False,
                                               apply_prefetcher=False)
    base_dataset = base_dataset.cache()
    base_dataset = base_dataset.prefetch(1)
    base_dataset = data_prefetcher.data_prefetcher(base_dataset, length=None,
                                                   permute_channel=False)
    scores = np.zeros(max_scans)

    base_gradient = calculate_model_gradient_vector(
        base_dataset, model, criterion, optimizer, args, num_batches)
    del base_dataset

    valid_set = set([1, 2, 5, 10])
    for i in range(1, max_scans + 1):
        if i not in valid_set:
            continue
        scan_dataset = \
            scan_view_dataset.get_scan_dataset(scan=i,
                                               image_size=args.image_size,
                                               batch_size=args.batch_size,
                                               permute=True,
                                               image_normalization=True,
                                               randomized_image_postprocessing=False,
                                               apply_prefetcher=False)
        scan_dataset = scan_dataset.prefetch(1)
        scan_dataset = data_prefetcher.data_prefetcher(scan_dataset, length=None,
                                                       permute_channel=False)
        scan_gradient = calculate_model_gradient_vector(
            scan_dataset, model, criterion, optimizer, args, num_batches)
        cosine_similarity = cosine_similarity_f(base_gradient,
                                                scan_gradient).cpu().numpy()
        scores[i - 1] = cosine_similarity
        tf.print("Scan gradient eval: {}={}".format(i, cosine_similarity))
    return scores

def get_train_loader(args):
    # Data loading code
    if not args.val_data_dir:
        traindir = os.path.join(args.data, 'train')
    else:
        traindir = args.data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.train_loader == "pytorch":
        if args.synthetic_loader:
            raise NotImplementedError(
                "Synthetic loader not implemented for PyTorch")
        else:
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(args.image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    else:
        train_sampler = None
        if args.imagenet_training:
            world_size = 1.0
            if args.world_size and args.world_size > 0.0:
                # Assume each worker gets batch_size batches to make a
                # macro-batch. Note that sharding makes observed datasets
                # different. World size is -1 by default, so this should be
                # ignored
                world_size = args.world_size
            dataset_size = math.ceil(1281167 / args.batch_size / world_size)
            shuffle_size = 10000
            cache_dataset = False
        else:
            dataset_size = None
            shuffle_size = 10000
            cache_dataset = False

        if args.multiprocessing_distributed:
            worker_index = args.local_rank
            world_size = args.world_size
            print("ENABLING SHARDING FOR MULTIPROCESSING"
                  " (worker {}/{})".format(worker_index, world_size))
        else:
            worker_index = None
            world_size = None

        def loader_fn(data_size):
            train_loader = loader_util.get_imagenet_tfdata_loader(
                args.data,
                args.data_format,
                args.scan,
                args.batch_size,
                shuffle_size=shuffle_size,
                dataset_size=data_size,
                cache_dataset=cache_dataset,
                permute=True,
                image_size=args.image_size,
                randomized_image_postprocessing=True,
                worker_index=worker_index,
                world_size=world_size,
                preprocessing_type=args.training_preprocessing,
            )
            return train_loader

        if args.synthetic_loader:
            if not dataset_size:
                train_loader = loader_fn(dataset_size)
                dataset_size = len(train_loader)
                del train_loader
            train_loader = \
                loader_util.get_synthetic_tfdata_loader(
                    batch_size=args.batch_size,
                    image_size=args.image_size,
                    dataset_size=dataset_size,
                    permute=True)
        else:
            train_loader = loader_fn(dataset_size)
            if not args.imagenet_training and args.world_size and args.world_size > 0.0:
                # Assume each worker gets batch_size batches to make a
                # macro-batch. Note that sharding makes observed datasets
                # different. World size is -1 by default, so this should be
                # ignored
                _world_size = args.world_size
                dataset_size = len(train_loader)
                dataset_size = math.ceil(dataset_size / _world_size)
                train_loader = loader_fn(dataset_size)

    return train_loader, train_sampler


def get_validation_loader(args):
    def expandpath(path_pattern):
        # https://stackoverflow.com/questions/51108256/how-to-take-a-pathname-string-with-wildcards-and-resolve-the-glob-with-pathlib
        p = pathlib.Path(path_pattern)
        return pathlib.Path(p.parent)
    # Data loading code
    if not args.val_data_dir:
        valdir = os.path.join(args.data, 'val')
    else:
        valdir = args.val_data_dir
    if "*" in valdir:
        # Glob path
        valdir_path = expandpath(valdir)
    else:
        valdir_path = pathlib.Path(valdir)
    if not valdir_path.exists():
        raise ValueError("Validation loader accessing directory '{}', which"
                         " does not exist".format(valdir_path))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.validation_loader == "pytorch":
        assert args.val_data_format == "files"
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        # NOTE(mkuchnik): This may give slightly different results due to being
        # `inception` style.
        if args.imagenet_training:
            dataset_size = math.ceil(50000 / args.batch_size)
            cache_dataset = False
        else:
            dataset_size = None
            cache_dataset = True
        if args.val_data_format == "TFRecord":
            basepath = valdir
            directory = None
        else:
            basepath = valdir_path.parent
            directory = valdir_path.name

        if args.distributed_validate:
            worker_index = args.local_rank
            world_size = args.world_size
            print("ENABLING VALIDATION SHARDING"
                  " (worker {}/{})".format(worker_index, world_size))
            # Unset dataset size to calculate shards
            dataset_size = None
        else:
            worker_index = None
            world_size = None

        val_loader = loader_util.get_imagenet_tfdata_loader(
            basepath,
            args.val_data_format,
            scan=0,
            batch_size=args.batch_size,
            shuffle_size=0,
            dataset_size=dataset_size,
            cache_dataset=cache_dataset,
            permute=True,
            image_size=224,
            randomized_image_postprocessing=False,
            directory=directory,
            worker_index=worker_index,
            world_size=world_size,
            deterministic=True,
        )
    return val_loader
