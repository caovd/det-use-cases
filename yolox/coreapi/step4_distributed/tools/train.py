#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.core import launch
from yolox.exp import Exp, check_exp_value, get_exp
from yolox.utils import configure_module, configure_nccl, configure_omp, get_num_devices

import determined as det

# New - Step 4 - Distributed
import os 
from yolox.utils import get_rank, get_local_rank, get_world_size

from datetime import timedelta
DEFAULT_TIMEOUT = timedelta(minutes=30)

def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="plz input your experiment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        nargs="?",
        const="ram",
        help="Caching imgs to ram/disk for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "-l",
        "--logger",
        type=str,
        help="Logger to be used for metrics. \
        Implemented loggers include `tensorboard` and `wandb`.",
        default="tensorboard"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(core_context, exp: Exp, args, trial_id, latest_checkpoint, hparams,
        # NEW - Step 4 - distributed 
        is_distributed, local_rank, rank, num_workers):    
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = exp.get_trainer(core_context, args, trial_id, latest_checkpoint, hparams,    
                            # NEW - Step 4 - distributed 
                            is_distributed, local_rank, rank, num_workers)    
    trainer.train()                                     

if __name__ == "__main__":

    info = det.get_cluster_info()
    
    # If running in local mode, cluster info will be None
    if info is not None:
        latest_checkpoint = info.latest_checkpoint
        trial_id = info.trial.trial_id
    else:
        latest_checkpoint = None
        trial_id = -1

    hparams = info.trial.hparams

    data_conf = info.user_data

    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(hparams, args.exp_file, args.name)
    exp.merge(args.opts)
    check_exp_value(exp)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    # New - Step 4 - Distributed
    # num_gpu = get_num_devices() if args.devices is None else args.devices
    # assert num_gpu <= get_num_devices()

    if args.cache is not None:
        exp.dataset = exp.get_dataset(cache=True, cache_type=args.cache)

    dist_url = "auto" if args.dist_url is None else args.dist_url

    # NEW - Step 4.2 - Defining distributed option for det.core.init()
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
        print("Distributed training")
    except:
        distributed = None
        print("No distributed training")
    
    #NEW - Step 4.1 - Check if we run in distributed training mode
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ['LOCAL_RANK']) # In lieu of get_local_rank()
        print("local rank: ", local_rank)
        
        is_distributed = True
        num_gpus = distributed.get_size()       
        dist_backend = hparams["dist_backend"]

        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend=dist_backend)
        torch.cuda.set_device(local_rank)
        torch.distributed.barrier()

        # Default values for num_machines and machine_rank
        num_machines = 1                # distributed.get_num_agents()
        machine_rank = hparams["machine_rank"]
        num_gpus_per_machine = num_gpus
        global_rank = machine_rank * num_gpus_per_machine + local_rank
        rank = global_rank
        logger.info("Rank {} initialization finished.".format(global_rank))
        torch.distributed.init_process_group(
                backend=dist_backend,
                init_method=dist_url,
                world_size=get_world_size(),
                rank=rank,
                timeout=DEFAULT_TIMEOUT,
        )
        
    else:
        print("No distributed training")
        local_rank = 0
        is_distributed = False
        num_gpus = 1
        dist_backend = None
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # print("Using " + str(device) + " to train")

    # New - Step 4 - distributed with det.core.init(distributed=distributed) 
    with det.core.init(distributed=distributed) as core_context:
        launch(
            main,   
            # New - Step 4 - Updated dist params
            num_gpus,
            num_machines,
            machine_rank,
            backend=dist_backend,
            dist_url=dist_url,
            args=(core_context, exp, args, trial_id, latest_checkpoint, hparams,  
                # NEW - Step 4 - distributed 
                is_distributed, local_rank, rank, num_gpus)
        )
