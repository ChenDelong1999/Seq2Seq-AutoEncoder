import argparse
import os
import json
import tqdm
import math
import numpy as np
import time
import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group, barrier
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

device = torch.device('cuda')


from data.dataset import get_dataset
from model import Seq2SeqAutoEncoderConfig, Seq2SeqAutoEncoderModel
from utils import ddp_setup, get_params_count_summary, save_hf_pretrained_model
from loss import seq2seq_autoencoder_loss

def train(model, dataloader, test_dataset, optimizer, scheduler, device, writer, epoch, args):

    model.train()
    step = epoch * len(dataloader)
    start_time_epoch = time.time()
    scaler = GradScaler()

    for i, (data, _) in enumerate(dataloader):
        start_time = time.time()
        data = data.to(device)

        with autocast():
            prediction = model(data)
            loss = seq2seq_autoencoder_loss(prediction, data, args.channel_info, args.num_queries)
            total_loss = sum(loss.values())
            total_loss = total_loss / args.gradient_accumulation_steps
            scaler.scale(total_loss).backward()

        if (step+1) % args.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if args.rank == 0:
                writer.add_scalar('train/grad_norm', grad_norm, step)

        if args.rank == 0:
            writer.add_scalar('train/loss (total)', total_loss.item() * args.gradient_accumulation_steps, step)
            for name, value in loss.items():
                writer.add_scalar(f'train/loss ({name})', value.item(), step)
                
            step_time = time.time() - start_time
            start_time = time.time()
            writer.add_scalar('train/step_time', step_time, step)

            writer.add_scalar('train/lr', scheduler.get_lr()[0], step)
            writer.add_scalar('train/data_seq_length_multiplier', dataloader.dataset.data_seq_length_multiplier, step)
            if step==0:
                print(f'Input data: {data.shape}\n{data}')

        if step % args.log_interval == 0 and args.rank == 0:
            log = f'Epoch {epoch+1}/{args.epochs}, Step {i}/{len(dataloader)} ({int(i/len(dataloader)*100)}%), Global Step {step}, '
            log += f'Loss: {total_loss.item() * args.gradient_accumulation_steps:.7f}, LR: {scheduler.get_lr()[0]:.7f}, '
            log += f'Step Time: {step_time:.3f}s'
            print(log)

        scheduler.step()
        if step < args.size_warmup_steps:
            multiplier = max(step / args.size_warmup_steps, args.size_warmup_min)
        else: 
            multiplier = 1
        dataloader.dataset.update_data_seq_length_multiplier(multiplier)
        step += 1

        if step!=0 and step % args.save_interval == 0 and args.rank == 0:   
            checkpoint_dir = os.path.join(args.checkpoint_dir, f'checkpoint_step{int((step+1)/1000)}k')
            save_hf_pretrained_model(model, checkpoint_dir)
            print(f'Saved checkpoint: {checkpoint_dir}')

    if args.rank == 0:
        print(f'Epoch {epoch+1}/{args.epochs} finished in {(time.time()-start_time_epoch)/60:.1f}min')

def main(rank, world_size, args):

    ddp_setup(rank, world_size, args.master_port)
    
    args.rank = rank

    if args.rank == 0:
        print(f'Loaded model config from {args.model_config}')
    if args.model_config.endswith('.json'):
        config = Seq2SeqAutoEncoderConfig.from_json_file(args.model_config)
        model = Seq2SeqAutoEncoderModel(config)
    else:
        model = Seq2SeqAutoEncoderModel.from_pretrained(args.model_config)
        if args.new_data_seq_length is not None:
            print(f'Changing model resolution. Original data_seq_length={model.config.data_seq_length}, new data_seq_length: {args.new_data_seq_length} (num_queries={model.config.num_queries})')
            model.change_resolution(args.new_data_seq_length)

    if args.rank == 0:
        print(model)
        print(model.config)
        print(get_params_count_summary(model))

    args.num_queries = model.config.num_queries
    args.data_seq_length = model.config.data_seq_length
    args.model_seq_length = model.config.model_seq_length
    
    # load dataset separately to avoid multiple downloads
    if args.rank == 0:
        train_dataset, test_dataset = get_dataset(args)
    if world_size > 1:
        barrier()

    if args.rank != 0:
        train_dataset, test_dataset = get_dataset(args)
    if world_size > 1:
        barrier()

    args.channel_info = train_dataset.channel_info

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size= args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        sampler=DistributedSampler(train_dataset, shuffle=True),
        prefetch_factor=8,
        )

    if args.torch_compile:
        model = torch.compile(model)
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=False)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    if args.scheduler == 'cosine':
        # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs*len(train_dataloader), eta_min=0)
        def lr_lambda(current_step: int):
            if current_step < args.lr_warmup_steps:
                return float(current_step) / float(max(1, args.lr_warmup_steps))
            else:
                return 0.5 * (1.0 + math.cos(math.pi * (current_step - args.lr_warmup_steps) / (len(train_dataloader) * args.epochs - args.lr_warmup_steps)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif args.scheduler == 'constant':
        def lr_lambda(current_step: int):
            if current_step < args.lr_warmup_steps:
                return float(current_step) / float(max(1, args.lr_warmup_steps))
            else:
                return 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif args.scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_dataloader), epochs=args.epochs, pct_start=0.05)

    if args.rank == 0:
        print('Arguments:')
        print('='*64)
        for k,v in vars(args).items():
            print(f'\t{k}: {v}')
        print('='*64)

        total_params = round(sum(p.numel() for p in model.parameters()) / 1e9, 3)
        total_params = f'{int(total_params*1000)}M' if total_params < 1 else f'{total_params:.1f}B'
        comment = f'-{args.dataset.upper()}-[{total_params}B-{args.num_queries}queries-{args.data_seq_length}]-[lr{args.lr}-bs{args.batch_size}x{args.gradient_accumulation_steps}step-{world_size}gpu]'
        if args.eval_only:
            comment += '-eval_only'
        writer = SummaryWriter(comment=comment)
        print(f'Writing logs to {writer.log_dir}')
        args.checkpoint_dir = os.path.join(writer.log_dir, 'checkpoints')
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        json.dump(vars(args), open(os.path.join(writer.log_dir, 'training_config.json'), 'w'), indent=4)
        json.dump(model.module.config.to_dict(), open(os.path.join(writer.log_dir, 'model_config.json'), 'w'), indent=4)
    else:
        writer = None

    for epoch in range(args.epochs):
        train_dataloader.sampler.set_epoch(epoch)
        train(model, train_dataloader, test_dataset, optimizer, scheduler, device, writer, epoch, args)

    if args.rank == 0:
        writer.close()
    destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Seq2Seq-AutoEncoder')
    parser.add_argument('--training_config', type=str, required=True, help='path to training config json file')
    parser.add_argument('--model_config', type=str, required=True, help='path to model config json file')
    parser.add_argument('--new_data_seq_length', type=int, default=None, help='high-resolution continuous pretraining')

    args = parser.parse_args()

    print(f'Loading training config from {args.training_config}')
    training_config = json.load(open(args.training_config))
    for k,v in training_config.items():
        setattr(args, k, v)

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args, ), nprocs=world_size)
