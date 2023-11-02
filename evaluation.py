import tqdm
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from loss import seq2seq_autoencoder_loss
from data.image_classification_dataset import decode_image_from_data

def evaluate(model, dataset, device, writer, step, args):

    model.eval()
    mse_sum = 0

    with torch.no_grad():

        targets = []
        reconstructions = []
        print('Runing conditional auto-regressive generation')
        for i in tqdm.tqdm(range(args.n_generation)):
            data, image_info = dataset[random.randint(0, len(dataset)-1)]
            data = data.to(device).unsqueeze(0)

            latents = model.module.encode(data)
            reconstructed = model.module.generate(latents)

            original_image, original_is_data, original_shape_encoding = decode_image_from_data(
                data.squeeze(0).cpu(), 
                image_info['width'], 
                image_info['height'], 
                dataset.num_queries, 
                img_channels=dataset.img_channels
                )
            reconstructed_image, reconstructed_is_data, reconstructed_shape_encoding = decode_image_from_data(
                reconstructed.squeeze(0).cpu(), 
                image_info['width'], 
                image_info['height'], 
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

            # log image pairs to tensorboard
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(original_image)
            ax[1].imshow(reconstructed_image)
            writer.add_figure(f'reconstruction/image_pair_{i}', fig, step)


            # mse_sum += ((data - reconstructed) ** 2).mean().item()
            targets.append(data.squeeze(0))
            reconstructions.append(reconstructed.squeeze(0))

        reconstructions = torch.stack(reconstructions)
        targets = torch.stack(targets)
        test_loss = seq2seq_autoencoder_loss(reconstructions, targets, args.channel_info)
        test_loss['total'] = sum(test_loss.values())

        # plot pixel intensity histograms of image pairs
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(10, 5)
        ax.hist(targets.cpu().numpy().flatten(), bins=100, alpha=0.5, label='original')
        ax.hist(reconstructions.cpu().numpy().flatten(), bins=100, alpha=0.5, label='reconstructed')
        ax.legend()
        writer.add_figure(f'reconstruction/histogram_{i}', fig, step)


        print('Starting extraction of latent representations and T-SNE')
        latents = []
        labels = []  
        for i in tqdm.tqdm(range(args.n_tsne_samples)):
            data, image_info = dataset[i]
            data = data.to(device).unsqueeze(0)
            latent = model.module.encode(data).flatten().cpu().numpy()
            latents.append(latent)
            labels.append(image_info['label'])

        latents = np.array(latents)
        labels = np.array(labels)

        tsne = TSNE(n_components=2, random_state=42)
        print('Running TSNE...') # this may take a while
        latents_tsne = tsne.fit_transform(latents)

        fig, ax = plt.subplots(1, 1)
        ax.scatter(latents_tsne[:, 0], latents_tsne[:, 1], c=labels, cmap='tab10', s=10, alpha=0.5)
        writer.add_figure(f'tsne/tsne', fig, step)

    return test_loss

