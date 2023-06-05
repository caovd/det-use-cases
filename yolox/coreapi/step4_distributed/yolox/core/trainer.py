#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.data import DataPrefetcher
from yolox.exp import Exp
from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    WandbLogger,
    adjust_status,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    mem_usage,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)

# NEW - import pathlib for defining checkpointing functions
import pathlib

# You can save multiple files, and use any file names or directory structures.
# All files nested under `checkpoint_directory` path will be included into the checkpoint.
def save_state(model, epoch, steps_completed, trial_id, checkpoint_directory):
    
    # Record some information, like number of epochs/batches completed and trial id in a "state" file
    with checkpoint_directory.joinpath("state").open("w") as f:
        f.write(f"{epoch}, {steps_completed}, {trial_id}")

    # NEW - Step 4 - distributed - take into account that you may want to save model.module state_dict if you run  distributed training
    model_to_save = model.module if hasattr(model, 'module') else model
    
    # Save model by reusing code from the original script, just changing the file name to account for checkpoint directory
    torch.save(model_to_save.state_dict(), checkpoint_directory.joinpath("yolox_s.pt"))

def load_state(model, trial_id, checkpoint_directory):
    checkpoint_directory = pathlib.Path(checkpoint_directory)
    with checkpoint_directory.joinpath("state").open("r") as f:
        epoch, steps_completed, ckpt_trial_id = [int(field) for field in f.read().split(",")]
    
    # Load model weights from the file saved using save_state function
    # NEW - Step 4 - if doing distributed training, load model using model.module.load_state_dict
    if hasattr(model, 'module'):
        model.module.load_state_dict(torch.load(checkpoint_directory.joinpath("yolox_s.pt")))
    else:
        model.load_state_dict(torch.load(checkpoint_directory.joinpath("yolox_s.pt")))
    
    # If the current trial id is the same as the id found in the checkpoint, then this is the continuation of a previously paused trial
    # so get the batch/epoch count back
    if ckpt_trial_id == trial_id:
        return model, epoch, steps_completed
    
    # Otherwise, this is a new trial; load the model weights but not the epoch/batch count.
    else:
        return model, 0, 0

class Trainer:
    def __init__(self, exp: Exp, core_context, args, trial_id, latest_checkpoint, hparams,
                 # NEW - Step 4 - distributed 
                is_distributed, local_rank, rank, num_gpus):  

        self.exp = exp    
        self.args = args
        self.hparams = hparams      # Unused
        
        self.core_context = core_context

        self.trial_id = trial_id
        self.latest_checkpoint = latest_checkpoint

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        
        # New - Step 4 - Distributed training - initation for additional params 
        # self.is_distributed = get_world_size() > 1 ==>
        self.is_distributed = is_distributed
        # self.rank = get_rank() ==>
        self.rank = local_rank
        # self.local_rank = get_local_rank() ==>
        self.local_rank = local_rank
        self.num_workers = num_gpus
        self.device = "cuda:{}".format(self.local_rank)      
        
        self.use_model_ema = exp.ema
        self.save_history_ckpt = exp.save_history_ckpt

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

        self.total_iters_processed = 0
        self.train_metrics = dict()
        self.val_metrics = dict()

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch() 
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        self.epoch = self.start_epoch
        self.last_checkpoint_epoch = -1
        self.last_eval_epoch = -1
        for op in self.core_context.searcher.operations():

            self.op = op
            while self.epoch < self.op.length:
                self.before_epoch()
                self.train_in_iter()
                self.after_epoch() 

                self.last_eval_epoch = self.epoch+1  
                                  
                # New - Step 4 - Only rank 0 reports progress
                if self.local_rank == 0:  
                    self.op.report_progress(self.epoch+1)

                self.last_checkpoint_epoch = self.epoch+1
                self.epoch += 1

            # New - Step 4 - Only rank 0 reports progress
            if self.local_rank == 0:
                if self.last_eval_epoch != self.epoch:
                    self.core_context.train.report_validation_metrics(
                        steps_completed=self.total_iters_processed, metrics={"AP50": self.ap50, "AP50_95": self.ap50_95}
                    )

            # New - Step 4 - Only rank 0 reports progress
            if self.local_rank == 0:
                self.op.report_completed(self.ap50)

        # New - Step 4 - Only rank 0 reports progress
        if self.local_rank == 0:
            if self.last_checkpoint_epoch != self.epoch:
                checkpoint_metadata = {"steps_completed": self.total_iters_processed}
                with self.core_context.checkpoint.store_path(checkpoint_metadata) as (path, uuid):
                    save_state(self.model, self.epoch, self.total_iters_processed, self.trial_id, path)
            
    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()                       

    def train_one_iter(self):
        iter_start_time = time.time()

        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, targets)

        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        logger.info(
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,

            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )
        # Tensorboard and Wandb loggers
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
            elif self.args.logger == "wandb":
                self.wandb_logger = WandbLogger.initialize_wandb_logger(
                    self.args,
                    self.exp,
                    self.evaluator.dataloader.dataset
                )
            else:
                raise ValueError("logger must be either 'tensorboard' or 'wandb'")

        logger.info("Training start...")
        logger.info("\n{}".format(model))

        # New - Step 4 - Only rank 0 reports progress
        if self.local_rank == 0:
            if self.latest_checkpoint is not None:
                print("Checkpoint provided, will load state")
                with self.core_context.checkpoint.restore_path(self.latest_checkpoint) as path:
                    self.model, self.start_epoch, self.tota_iters_processed = load_state(model, self.trial_id, path)
                print("Successfully loaded checkpoint")
                if self.start_epoch == 0:
                    print("Will start training the model as part as the new trial " + str(self.trial_id))
                else:
                    print("Continuation of trial " + str(self.trial_id) + " from epoch " + str(self.start_epoch) + " after training for " + str(self.total_iters_processed) + " steps")

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        )
        if self.rank == 0:
            if self.args.logger == "wandb":
                self.wandb_logger.finish()

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True
            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            mem_str = "gpu mem: {:.0f}Mb, mem: {:.1f}Gb".format(gpu_mem_usage(), mem_usage())

            self.total_iters_processed += self.iter + 1
            for k, v in loss_meter.items():
                self.train_metrics[k] = v.latest.item()        # item() to convert tensor to scalar

            # New - Step 4 - All reduce loss
            metrics_keys = ["total_loss", "l1_loss", "iou_loss", "conf_loss", "cls_loss"]
            self.total_loss, self.l1_loss, self.iou_loss, self.conf_loss, self.cls_loss = [self.train_metrics[v] for v in metrics_keys]
            if self.is_distributed:
                self.reduced_metrics = dict()
                for k in metrics_keys:
                    tensor_loss = torch.tensor([self.train_metrics[k]], dtype=torch.float64)
                    tensor_loss = tensor_loss.to(self.device)
                    torch.distributed.all_reduce(tensor_loss, op=torch.distributed.ReduceOp.SUM)
                    self.reduced_metrics[k] = tensor_loss.item() / self.num_workers

                # Update the reduced metrics
                self.metrics = self.reduced_metrics

            # New - Step 4 - Only rank 0 reports progress
            if self.local_rank == 0:
                self.core_context.train.report_training_metrics(
                    steps_completed=self.total_iters_processed, metrics=self.train_metrics
                    )

            logger.info(
                "{}, {}, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    mem_str,
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )

            if self.rank == 0:
                if self.args.logger == "wandb":
                    metrics = {"train/" + k: v.latest for k, v in loss_meter.items()}
                    metrics.update({
                        "train/lr": self.meter["lr"].latest
                    })
                    self.wandb_logger.log_metrics(metrics, step=self.progress_in_iter)

            self.meter.clear_meters()

        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_ap = ckpt.pop("best_ap", 0)
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        with adjust_status(evalmodel, training=False):
            (ap50_95, ap50, summary), predictions = self.exp.eval(
                evalmodel, self.evaluator, self.is_distributed, return_outputs=True
            )

        update_best_ckpt = ap50_95 > self.best_ap
        self.best_ap = max(self.best_ap, ap50_95)

        # New - Step 4 - Only rank 0 reports progress
        if self.local_rank == 0:
            if (self.iter + 1) % self.exp.print_interval == 0:
                self.val_metrics["AP50"] = ap50
                self.val_metrics["AP50_95"] = ap50_95
                self.core_context.train.report_validation_metrics(
                    steps_completed=self.total_iters_processed, metrics=self.val_metrics
                    )

        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
                self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            if self.args.logger == "wandb":
                self.wandb_logger.log_metrics({
                    "val/COCOAP50": ap50,
                    "val/COCOAP50_95": ap50_95,
                    "train/epoch": self.epoch + 1,
                })
                self.wandb_logger.log_images(predictions)
            logger.info("\n" + summary)
        synchronize()

        self.save_ckpt("last_epoch", update_best_ckpt, ap=ap50_95)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}", ap=ap50_95)

        self.ap50 = ap50
        self.ap50_95 = ap50_95
        
        # New - Step 4 - Only rank 0 reports progress
        if self.local_rank == 0:
            self.checkpoint_metadata = {"steps_completed": self.total_iters_processed}
            with self.core_context.checkpoint.store_path(self.checkpoint_metadata) as (path, uuid):
                save_state(self.model, self.epoch+1, self.total_iters_processed, self.trial_id, path)
         
        if self.core_context.preempt.should_preempt():
            print("Preemption signal detected, will stop training")
            # At this point, a checkpoint ws just saved, so training can exit
            # immediately and resume when the trial is reactivated.

        return 

    def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_ap": self.best_ap,
                "curr_ap": ap,
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )

            if self.args.logger == "wandb":
                self.wandb_logger.save_checkpoint(
                    self.file_name,
                    ckpt_name,
                    update_best_ckpt,
                    metadata={
                        "epoch": self.epoch + 1,
                        "optimizer": self.optimizer.state_dict(),
                        "best_ap": self.best_ap,
                        "curr_ap": ap
                    }
                )
