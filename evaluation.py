import tqdm
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from loss import seq2seq_autoencoder_loss
from PIL import Image

def evaluate(model, dataset, device, writer, step, args):

    model.eval()
    mse_sum = 0

    with torch.no_grad():

        targets = []
        reconstructions = []
        print('Runing conditional auto-regressive generation')
        for i in tqdm.tqdm(range(args.n_generation)):
            data, sample_info = dataset[random.randint(0, len(dataset)-1)]
            data = data.to(device).unsqueeze(0)

            latents = model.module.encode(data)
            reconstructed = model.module.generate(latents)

            original_segment, original_is_data, original_shape_encoding = dataset.decode_image_from_data(
                data.squeeze(0).cpu(), 
                sample_info['width'], 
                sample_info['height'], 
                dataset.num_queries, 
                img_channels=dataset.img_channels
                )
            reconstructed_segment, reconstructed_is_data, reconstructed_shape_encoding = dataset.decode_image_from_data(
                reconstructed.squeeze(0).cpu(), 
                sample_info['width'], 
                sample_info['height'], 
                dataset.num_queries, 
                img_channels=dataset.img_channels
                )

            # plot original and reconstructed shape_encoding lines
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(15, 3)
            ax.plot(original_shape_encoding, label='original')
            ax.plot(reconstructed_shape_encoding, label='reconstructed')
            ax.legend()
            writer.add_figure(f'special_tokens/shape_encoding_{i}', fig, step)
            
            # plot is_data lines
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(15, 3)
            ax.plot(original_is_data, label='original')
            ax.plot(reconstructed_is_data, label='reconstructed')
            ax.legend()
            writer.add_figure(f'special_tokens/is_data_{i}', fig, step)

            # plot segment pairs
            fig = visualize_segments(sample_info, original_segment, reconstructed_segment)
            writer.add_figure(f'reconstruction/image_pair_{i}', fig, step)

            targets.append(data.squeeze(0))
            reconstructions.append(reconstructed.squeeze(0))

        reconstructions = torch.stack(reconstructions)
        targets = torch.stack(targets)
        test_loss = seq2seq_autoencoder_loss(reconstructions, targets, args.channel_info)
        test_loss['total'] = sum(test_loss.values())

        print('Starting extraction of latent representations and T-SNE')
        latents = []
        labels = []  
        for i in tqdm.tqdm(range(args.n_tsne_samples)):
            data, sample_info = dataset[i]
            data = data.to(device).unsqueeze(0)
            latent = model.module.encode(data).flatten().cpu().numpy()
            latents.append(latent)
            labels.append(sample_info['label'])

        latents = np.array(latents)
        labels = np.array(labels)

        tsne = TSNE(n_components=2, random_state=42)
        print('Running TSNE...') # this may take a while
        latents_tsne = tsne.fit_transform(latents)

        fig, ax = plt.subplots(1, 1)
        ax.scatter(latents_tsne[:, 0], latents_tsne[:, 1], c=labels, cmap='tab10', s=10, alpha=0.5)
        writer.add_figure(f'tsne/tsne', fig, step)

    return test_loss


def visualize_segments(sample_info, original_segment, reconstructed_segment):

    if 'image_path' in sample_info.keys():
        fig, ax = plt.subplots(1, 3)
        fig.set_size_inches(15, 5)
        ax[0].imshow(Image.open(sample_info['image_path']))
        x, y, w, h = sample_info['bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, color='red')
        ax[0].add_patch(rect)
        ax[0].axis('off')
        
        ax[1].imshow(original_segment)
        ax[2].imshow(reconstructed_segment)
        ax[2].axis('off')
    else:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(original_segment)
        ax[1].imshow(reconstructed_segment)

    ax[0].axis('off')
    ax[1].axis('off')

    return fig