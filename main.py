import argparse
import os
import tqdm
import numpy as np
import time
import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter


from data import get_dataset
from model import Seq2SeqAutoEncoderConfig, Seq2SeqAutoEncoderModel
from utils import get_params_count_summary, save_hf_pretrained_model
from evaluate import evaluate

def ddp_setup(rank: int, world_size: int):
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  init_process_group(backend="nccl", rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)


def train(model, dataloader, test_loader, optimizer, scheduler, device, writer, epoch, args):

    model.train()
    step = epoch * len(dataloader)
    start_time = time.time()
    for i, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        loss =  model(data)['loss']
        loss.backward()

        optimizer.step()
        if args.rank == 0:
            writer.add_scalar('train/loss', loss.item(), step)
            writer.add_scalar('train/lr', scheduler.get_lr()[0], step)

        if i % args.log_interval == 0 and args.rank == 0:
            step_time = (time.time() - start_time) / args.log_interval
            start_time = time.time()
            writer.add_scalar('train/step_time', step_time, step)
            print(f'Epoch {epoch+1}/{args.epochs}, Step {i}/{len(dataloader)} ({int(i/len(dataloader)*100)}%), Global Step {step},\tLoss: {loss.item():.7f}, LR: {scheduler.get_lr()[0]:.7f},\tStep Time: {step_time:.3f}s')

        scheduler.step()
        step += 1

        if step!=0 and step % args.save_interval == 0 and args.rank == 0:   
            checkpoint_dir = os.path.join(args.checkpoint_dir, f'checkpoint_ep{epoch}_step{i+1}')
            save_hf_pretrained_model(model, checkpoint_dir)
            print(f'Saved checkpoint: {checkpoint_dir}')
            
        if step!=0 and step % args.eval_interval == 0 and args.rank == 0:
            print('Start evaluations')
            test_mse = evaluate(model, test_loader, device, writer, step, args)
            writer.add_scalar('train/test_generation_mse', test_mse, step)
            print(f'Test Generation MSE: {test_mse:.7f}')


def main(rank, world_size, args):

    ddp_setup(rank, world_size)
    
    args.rank = rank

    if args.rank == 0:
        train_dataset, test_dataset = get_dataset(args)
    if world_size > 1:
        barrier()

    if args.rank != 0:
        train_dataset, test_dataset = get_dataset(args)
    if world_size > 1:
        barrier()

    args.max_seq_length = train_dataset.max_seq_length
    model_seq_length = args.max_seq_length + args.num_queries + 1 # additional one starting token

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size= args.batch_size,
        shuffle=False, 
        num_workers=8,
        sampler=DistributedSampler(train_dataset, shuffle=True)
        )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size= args.batch_size,
        shuffle=False, 
        num_workers=8,
        )

    # Set up the model and optimizer
    config = Seq2SeqAutoEncoderConfig(
        prediction_length=model_seq_length,
        context_length=model_seq_length,
        input_size=train_dataset.num_channels,
        num_time_features=1,
        lags_sequence=[0],
        scaling="",
        distribution_output="non_probabilistic",
        loss="mse",

        # transformer params:
        d_model=args.d_model,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        encoder_attention_heads=args.encoder_attention_heads,
        decoder_attention_heads=args.decoder_attention_heads,
        encoder_ffn_dim=args.encoder_ffn_dim,
        decoder_ffn_dim=args.decoder_ffn_dim,

        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,

        num_queries=args.num_queries,
        d_latent=args.d_latent
    )

    device = torch.device('cuda')

    if args.pretrained is not None:
        model = Seq2SeqAutoEncoderModel.from_pretrained(args.pretrained)
    else:
        model = Seq2SeqAutoEncoderModel(config)

    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=False)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_dataloader), epochs=args.epochs, pct_start=0.05)

    # Set up TensorBoard logging
    if args.rank == 0:

        print(model)
        print(model.module.config)
        print(get_params_count_summary(model))

        print('Arguments:')
        print('='*64)
        for k,v in vars(args).items():
            print(f'\t{k}: {v}')
        print('='*64)

        total_params = round(sum(p.numel() for p in model.parameters()) / 1e9 ,2)
        comment = f'-{args.dataset}-[model={total_params}B]-[lr{args.lr}-bs{args.batch_size}-ngpu{world_size}]'
        if args.eval_only:
            comment += '-eval_only'
        writer = SummaryWriter(comment=comment)
        print(f'Writing logs to {writer.log_dir}')
        args.checkpoint_dir = os.path.join(writer.log_dir, 'checkpoints')
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    else:
        writer = None

    if not args.eval_only:
        for epoch in range(args.epochs):
            train_dataloader.sampler.set_epoch(epoch)
            train(model, train_dataloader, test_dataloader, optimizer, scheduler, device, writer, epoch, args)

    elif args.rank == 0:
        test_mse = evaluate(model, test_dataloader, device, writer, step=0, args=args)
        writer.add_scalar('test/mse', test_mse, 0)
        print(f'Test MSE: {test_mse:.7f}')

    if args.rank == 0:
        writer.close()

    destroy_process_group()


if __name__ == '__main__':
    # Define command line arguments
    parser = argparse.ArgumentParser(description='Seq2Seq-AutoEncoder')
    # parser.add_argument('--max_seq_length', type=int, default=1024, help='maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')

    # Data parameters
    parser.add_argument('--dataset', type=str, default='CIFAR100')
    parser.add_argument('--data_dir', type=str, default='dataset', help='path to dataset directory')

    # Model parameters
    parser.add_argument('--d_model', type=int, default=512, help='dimensionality of the model')
    parser.add_argument('--encoder_layers', type=int, default=8, help='number of encoder layers')
    parser.add_argument('--decoder_layers', type=int, default=8, help='number of decoder layers')
    parser.add_argument('--encoder_attention_heads', type=int, default=8, help='number of encoder attention heads')
    parser.add_argument('--decoder_attention_heads', type=int, default=8, help='number of decoder attention heads')
    parser.add_argument('--encoder_ffn_dim', type=int, default=1024, help='dimensionality of the encoder feedforward network')
    parser.add_argument('--decoder_ffn_dim', type=int, default=1024, help='dimensionality of the decoder feedforward network')

    parser.add_argument('--num_queries', type=int, default=64, help='number of queries')
    parser.add_argument('--d_latent', type=int, default=512, help='dimensionality of the latent space')

    # Training parameters
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--log_interval', type=int, default=100, help='number of steps between each logging')
    parser.add_argument('--save_interval', type=int, default=1000, help='number of epochs between each checkpoint saving')
    parser.add_argument('--eval_interval', type=int, default=1000, help='number of epochs between each checkpoint saving')
    parser.add_argument('--pretrained', type=str, default=None, help='path to checkpoint')

    # Evaluation parameters
    parser.add_argument('--eval_only', action='store_true', help='evaluate the model on the test set without training')
    parser.add_argument('--n_generation', type=int, default=5, help='number of generations to run during evaluation')
    parser.add_argument('--n_tsne_samples', type=int, default=2000, help='number of samples to run T-SNE on during evaluation')

    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args, ), nprocs=world_size)

"""

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python main.py \
    --dataset stl10 \
    --log_interval=100 --eval_interval 2500 --save_interval=10000 \
    --batch_size=3 --lr=1e-5  --n_generation=1 \
    --d_model 1024 --encoder_layers 12 --decoder_layers 4 \
    --encoder_attention_heads 8 --decoder_attention_heads 8 \
    --encoder_ffn_dim 1024 --decoder_ffn_dim 1024 \
    --num_queries 64 --d_latent 2048
    
"""