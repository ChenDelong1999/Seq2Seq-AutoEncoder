import argparse
import os
import tqdm
import numpy as np

import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from data import SeqImgClsDataset, decode_image_from_data
from model import Seq2SeqAutoEncoderConfig, Seq2SeqAutoEncoderModel
from utils import get_params_count_summary


def ddp_setup(rank: int, world_size: int):
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  init_process_group(backend="nccl", rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)


def train(model, dataloader, test_loader, optimizer, scheduler, device, writer, epoch, args):

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
            
            print('Start evaluations')
            test_mse = evaluate(model, test_loader, device, writer, step, args)
            writer.add_scalar('train/test_generation_mse', test_mse, step)
            print(f'Test Generation MSE: {test_mse:.7f}')


def evaluate(model, dataloader, device, writer, step, args):

    model.eval()
    mse_sum = 0

    dataset = dataloader.dataset
    with torch.no_grad():

        print('Starting extraction of latent representations and T-SNE')
        latents = []
        labels = []  
        for i in tqdm.tqdm(range(args.n_tsne_samples)):
            data, label = dataset[i]
            data = data.to(device).unsqueeze(0)
            latent = model.module.encode(data).flatten().cpu().numpy()
            latents.append(latent)
            labels.append(label['class'])

        latents = np.array(latents)
        labels = np.array(labels)

        tsne = TSNE(n_components=2, random_state=42)
        print('Running TSNE...') # this may take a while
        latents_tsne = tsne.fit_transform(latents)

        fig, ax = plt.subplots(1, 1)
        ax.scatter(latents_tsne[:, 0], latents_tsne[:, 1], c=labels, cmap='tab10', s=10, alpha=0.5)
        writer.add_figure(f'tsne/tsne', fig, step)


        print('Runing conditional auto-regressive generation')
        for i in tqdm.tqdm(range(args.n_generation)):
            data, label = dataset[i]
            data = data.to(device).unsqueeze(0)

            latents = model.module.encode(data)
            reconstructed = model.module.generate(latents)

            original_image, _ = decode_image_from_data(data.squeeze(0).cpu(), label['width'], label['height'], dataset.num_queries)
            reconstructed_image, _ = decode_image_from_data(reconstructed.squeeze(0).cpu(), label['width'], label['height'], dataset.num_queries)

            # log image pairs to tensorboard
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(original_image)
            ax[1].imshow(reconstructed_image)
            writer.add_figure(f'generation/image_pair_{i}', fig, step)
            
            mse_sum += ((data - reconstructed) ** 2).mean().item()

    return mse_sum / len(dataset)


def main(rank, world_size, args):

    ddp_setup(rank, world_size)
    
    args.rank = rank
    model_seq_length = args.max_seq_length + args.num_queries + 1 # additional one starting token

    train_dataset = SeqImgClsDataset(
        dataset=CIFAR100(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor()),
        max_seq_length=args.max_seq_length,
        num_queries=args.num_queries,
    )
    test_dataset = SeqImgClsDataset(
        dataset=CIFAR100(root=args.data_dir, train=False, download=True, transform=transforms.ToTensor()),
        max_seq_length=args.max_seq_length,
        num_queries=args.num_queries,
    )

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

    model = Seq2SeqAutoEncoderModel(config).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    if args.pretrained is not None:
        message = model.load_state_dict(torch.load(args.pretrained, map_location=device), strict=False)
        print(f'Loaded pretrained weights from {args.pretrained}: {message}')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_dataloader))

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

        comment = f'-eval_only={args.eval_only}'
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
    parser = argparse.ArgumentParser(description='Train a time series transformer model on CIFAR100 dataset.')
    parser.add_argument('--data_dir', type=str, default='dataset', help='path to dataset directory')
    parser.add_argument('--max_seq_length', type=int, default=1024, help='maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')

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
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--log_interval', type=int, default=100, help='number of steps between each logging')
    parser.add_argument('--save_interval', type=int, default=1000, help='number of epochs between each checkpoint saving')
    parser.add_argument('--pretrained', type=str, default=None, help='path to checkpoint')

    # Evaluation parameters
    parser.add_argument('--eval_only', action='store_true', help='evaluate the model on the test set without training')
    parser.add_argument('--n_generation', type=int, default=5, help='number of generations to run during evaluation')
    parser.add_argument('--n_tsne_samples', type=int, default=1000, help='number of samples to run T-SNE on during evaluation')

    args = parser.parse_args()


    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args, ), nprocs=world_size)
