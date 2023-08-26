# coding=utf-8
"""
Code for model constructing, evaluations, and distributed training frameworks comes from:
https://github.com/minghchen/CARL_code
Please request the consent of the original authors for any commercial use.
"""
import os
from posixpath import split
import sys
import pprint
import torch
import random
from tqdm import tqdm
import numpy as np

from utils.parser import parse_args, load_config, setup_train_dir
from models import build_model, save_checkpoint, load_checkpoint
from utils.optimizer import construct_optimizer, construct_scheduler, get_lr
from datasets import construct_dataloader, unnorm
from evaluation import get_tasks
from VSP import PCL

import utils.distributed as dist
import utils.logging as logging
from torch.utils.tensorboard import SummaryWriter


logger = logging.get_logger(__name__)

def train(cfg, train_loader, model, optimizer, scheduler, algo, cur_epoch,summary_writer):
    model.train()
    optimizer.zero_grad()
    data_size = len(train_loader)
    # DistributedSampler shuffle based on epoch and seed
    if hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(cur_epoch)
        logger.info(f"update the training sampler to epoch {cur_epoch}")
    if hasattr(train_loader.batch_sampler, 'set_epoch'):
        train_loader.batch_sampler.set_epoch(cur_epoch)
        logger.info(f"update the training batch sampler to epoch {cur_epoch}")

    # dist.init_process_group(backend='gloo')
    if dist.is_root_proc():
        train_loader = tqdm(train_loader, total=len(train_loader))
    for cur_iter, (videos, labels, seq_lens, bridges, names, ids) in enumerate(train_loader):
        optimizer.zero_grad()
        if cfg.USE_AMP:
            torch.autograd.set_detect_anomaly(True)
            scaler = algo.scaler
            with torch.cuda.amp.autocast():
                loss = algo.vsp_loss(model, videos, bridges, labels, names)
            scaler.scale(loss).backward()
            if cfg.OPTIMIZER.GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIMIZER.GRAD_CLIP)
            scaler.step(optimizer)
            # Updates the scale for next iteration.
            scaler.update()
        else:
            loss = algo.vsp_loss(model, videos, bridges, labels, names)
            # Perform the backward pass.
            loss.backward()
            # Update the parameters.
            if cfg.OPTIMIZER.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIMIZER.GRAD_CLIP)
            optimizer.step()

        dist.all_reduce(loss)[0].item() / data_size

    summary_writer.add_scalar('train/learning_rate', get_lr(optimizer)[0], cur_epoch)
    summary_writer.add_scalar(f'train/loss', loss, cur_epoch)
    logger.info("epoch {}, train loss: {:.3f}".format(cur_epoch, loss))
    
    if cur_epoch != cfg.TRAIN.MAX_EPOCHS-1:
        scheduler.step()


def val(cfg, val_loader, model, algo, cur_epoch, summary_writer):
    model.eval()
    data_size = len(val_loader)

    with torch.no_grad():
        for cur_iter, (videos, labels, seq_lens, bridges, names, ids) in enumerate(val_loader):
            if cfg.USE_AMP:
                with torch.cuda.amp.autocast():   
                    loss = algo.vsp_loss(model, videos, bridges, labels, names)
            else:
                loss = algo.vsp_loss(model, videos, bridges, labels, names)
            dist.all_reduce(loss)[0].item() / data_size
    summary_writer.add_scalar(f'train/loss', loss, cur_epoch)
    logger.info("epoch {}, train loss: {:.3f}".format(cur_epoch, loss))



def main():
    args = parse_args()
    cfg = load_config(args)

    setup_train_dir(cfg, cfg.LOGDIR, args.continue_train)
    cfg.PATH_TO_DATASET = os.path.join(args.workdir, cfg.PATH_TO_DATASET)
    cfg.NUM_GPUS = torch.cuda.device_count() # num_gpus_per_machine

    args.world_size = int(os.getenv('WORLD_SIZE')) # total_gpus
    if os.environ.get('OMPI_COMM_WORLD_SIZE') is None:
        args.rank = args.local_rank
    else:
        args.node_rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
        args.rank = args.node_rank * torch.cuda.device_count() + args.local_rank
    logger.info(f'Node info: rank {args.rank} of world size {args.world_size}')
    cfg.args = args



    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    dist.init_distributed_training(cfg)
    # Set random seed from configs.
    random.seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # distributed logging and ignore warning message
    logging.setup_logging(cfg.LOGDIR)
    summary_writer = SummaryWriter(os.path.join(cfg.LOGDIR, 'train_logs'))

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the model for parallel
    model = build_model(cfg)
    torch.cuda.set_device(args.local_rank)
    model = model.cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [args.local_rank], 
            output_device = args.local_rank, find_unused_parameters=True)

    optimizer = construct_optimizer(model, cfg)
    algo = PCL(cfg)

    # Setup Dataset Iterators from train and val datasets.
    train_loader, train_emb_loader = construct_dataloader(cfg, "train")
    val_loader, val_emb_loader = construct_dataloader(cfg, "val")
    iterator_tasks, embedding_tasks = get_tasks(cfg)

    if cfg.USE_AMP:
        algo.scaler = torch.cuda.amp.GradScaler()
        logger.info("Initializing mixed precision done.")

    """Trains model and evaluates on relevant downstream tasks."""
    start_epoch = load_checkpoint(cfg, model, optimizer)
    cfg.TRAIN.MAX_ITERS = cfg.TRAIN.MAX_EPOCHS * len(train_loader)
    scheduler = construct_scheduler(optimizer, cfg)

    for cur_epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCHS):
        logger.info(f"Traning epoch {cur_epoch}/{cfg.TRAIN.MAX_EPOCHS}, {len(train_loader)} iters each epoch")
        train(cfg, train_loader, model, optimizer, scheduler, algo, cur_epoch, summary_writer)
        if (cur_epoch+1) % cfg.EVAL.VAL_INTERVAL == 0 or cur_epoch == cfg.TRAIN.MAX_EPOCHS-1:
            val(cfg, val_loader, model, algo, cur_epoch, summary_writer)
            if cfg.DATASETS[0] == "finegym":
                from evaluate_finegym import evaluate_once
                evaluate_once(cfg, model, train_loader, val_loader, train_emb_loader, val_emb_loader, 
                                iterator_tasks, embedding_tasks, cur_epoch, summary_writer)
            elif dist.is_root_proc():
                from evaluate import evaluate_once
                evaluate_once(cfg, model, train_loader, val_loader, train_emb_loader, val_emb_loader, 
                                iterator_tasks, embedding_tasks, cur_epoch, summary_writer)
        if dist.is_root_proc() and ((cur_epoch+1) % cfg.CHECKPOINT.SAVE_INTERVAL == 0 or cur_epoch == cfg.TRAIN.MAX_EPOCHS-1):
            save_checkpoint(cfg, model, optimizer, cur_epoch)
        dist.synchronize()

    torch.distributed.destroy_process_group()
if __name__ == '__main__':
    main()
