import argparse
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import Seq2SeqAutoEncoderConfig, Seq2SeqAutoEncoderModel
import argparse
import logging

torch.backends.cudnn.benchmark = True

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data import SeqImgClsDataset, decode_image_from_data
from utils import get_params_count_summary
import os

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

def ddp_setup(rank: int, world_size: int):
  """
  Args:
      rank: Unique identifier of each process
     world_size: Total number of processes
  """
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  init_process_group(backend="nccl", rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)


def train(model, dataloader, optimizer, scheduler, device, writer, epoch, args):

    model.train()
    step = epoch * len(dataloader)
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
            print(f'Epoch {epoch+1}/{args.epochs}, Stpe {i}/{len(dataloader)}, Global Step {step}\tLoss: {loss.item():.7f}, LR: {scheduler.get_lr()[0]:.7f}')

        scheduler.step()
        step += 1

        if i!=0 and i % args.save_interval == 0 and args.rank == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_ep{epoch}_step{i}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Saved checkpoint: {checkpoint_path}')
            
            # Evaluate the model on the test set
            test_mse = evaluate(model, dataloader.dataset, device, writer, step)
            writer.add_scalar('test/train_generation_mse', test_mse, step)
            print(f'Train Generation MSE: {test_mse:.7f}')


def evaluate(model, dataset, device, writer, step):

    model.eval()
    mse_sum = 0
    with torch.no_grad():
        for i in range(5):
            data, label = dataset[i]
            data = data.to(device).unsqueeze(0)

            latents = model.module.encode(data)
            reconstructed = model.module.generate(latents)

            original_image, _ = decode_image_from_data(data.squeeze(0).cpu(), label['width'], label['height'], dataset.num_queries)
            reconstructed_image, _ = decode_image_from_data(reconstructed.squeeze(0).cpu(), label['width'], label['height'], dataset.num_queries)

            # log image pairs to tensorboard
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(original_image)
            ax[1].imshow(reconstructed_image)
            writer.add_figure(f'test/image_pair_{i}', fig, step)
            
            mse_sum += ((data - reconstructed) ** 2).mean().item()

    return mse_sum / len(dataset)


def main(rank, world_size, args):

    ddp_setup(rank, world_size)
    
    args.rank = rank

    # Set up the dataset and dataloader
    max_seq_length = args.max_seq_length
    num_queries = args.num_queries
    batch_size = args.batch_size
    model_seq_length = max_seq_length + num_queries + 1

    train_dataset = SeqImgClsDataset(
        dataset=torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor()),
        max_seq_length=max_seq_length,
        num_queries=num_queries,
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=8,
        sampler=DistributedSampler(train_dataset, shuffle=True)
        )

    test_dataset = SeqImgClsDataset(
        dataset=torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transforms.ToTensor()),
        max_seq_length=max_seq_length,
        num_queries=num_queries,
    )
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

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

        num_queries=num_queries,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = Seq2SeqAutoEncoderModel(config).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if args.pretrained is not None:
        message = model.load_state_dict(torch.load(args.pretrained, map_location=device))
        print(f'Loaded pretrained weights from {args.pretrained}: {message}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_dataloader))

    # Set up TensorBoard logging
    if args.rank == 0:

        print(model)
        print(model.module.config)
        print(get_params_count_summary(model))

        print('Arguments:')
        print('='*32)
        for k,v in vars(args).items():
            print(f'\t{k}: {v}')
        print('='*32)

        writer = SummaryWriter()
        print(f'Writing logs to {writer.log_dir}')
        args.checkpoint_dir = os.path.join(writer.log_dir, 'checkpoints')
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    else:
        writer = None


    if not args.eval_only:
        # Train the model for the specified number of epochs
        for epoch in range(args.epochs):
            train_dataloader.sampler.set_epoch(epoch)
            train(model, train_dataloader, optimizer, scheduler, device, writer, epoch, args)

    elif args.rank == 0:
        # Evaluate the model on the test set
        test_mse = evaluate(model, test_dataset, device, writer, step=0)
        writer.add_scalar('test/mse', test_mse, 0)
        print(f'Test MSE: {test_mse:.7f}')
    writer.close()

    destroy_process_group()


if __name__ == '__main__':
    # Define command line arguments
    parser = argparse.ArgumentParser(description='Train a time series transformer model on CIFAR100 dataset.')
    parser.add_argument('--data_dir', type=str, default='dataset', help='path to dataset directory')
    parser.add_argument('--max_seq_length', type=int, default=1024, help='maximum sequence length')
    parser.add_argument('--num_queries', type=int, default=64, help='number of queries')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--eval_only', action='store_true', help='evaluate the model on the test set without training')

    # Model parameters
    parser.add_argument('--d_model', type=int, default=512, help='dimensionality of the model')
    parser.add_argument('--encoder_layers', type=int, default=8, help='number of encoder layers')
    parser.add_argument('--decoder_layers', type=int, default=8, help='number of decoder layers')
    parser.add_argument('--encoder_attention_heads', type=int, default=8, help='number of encoder attention heads')
    parser.add_argument('--decoder_attention_heads', type=int, default=8, help='number of decoder attention heads')
    parser.add_argument('--encoder_ffn_dim', type=int, default=1024, help='dimensionality of the encoder feedforward network')
    parser.add_argument('--decoder_ffn_dim', type=int, default=1024, help='dimensionality of the decoder feedforward network')

    # Training parameters
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--log_interval', type=int, default=100, help='number of steps between each logging')
    parser.add_argument('--save_interval', type=int, default=1000, help='number of epochs between each checkpoint saving')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='directory to save checkpoints in')
    parser.add_argument('--pretrained', type=str, default=None, help='path to checkpoint')
    args = parser.parse_args()


    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args, ), nprocs=world_size)
