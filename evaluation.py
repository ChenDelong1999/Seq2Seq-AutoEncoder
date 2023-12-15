import argparse
import os
import json
import tqdm
import math
import numpy as np
import random
import time
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image
import pprint
import pandas as pd
import argparse
import os

from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors

np.random.seed(42)
random.seed(42)

from data.dataset import get_dataset, SeqMaskDataset, LVISDataset, V3DetDataset, COCODataset, VisualGenomeDataset, SA1BDataset
from model import Seq2SeqAutoEncoderModel, Seq2SeqAutoEncoderConfig


def decode_image_from_seq(data_seq, new_line_threshold=0.5, mask_threshold=0.5):
    # input: seq (tensor of shape (seq_length, 5))

    # first position is <start-of-sequence> token and should be ignored
    rgb_seq = data_seq[1:, :3]
    new_line_seq = data_seq[1:, 3] > new_line_threshold
    mask_seq = data_seq[1:, 4] > mask_threshold

    # find the last positive element in new_line_seq, and use it to truncate data and sequences
    if np.sum(new_line_seq) > 0:
        effective_seq_length = np.nonzero(new_line_seq)[0][-1] + 1 # +1 because we ignored the first token 
    else:
        effective_seq_length = len(new_line_seq)

    rgb_seq = rgb_seq[:effective_seq_length] * 255
    new_line_seq = new_line_seq[:effective_seq_length]
    mask_seq = mask_seq[:effective_seq_length]

    if np.sum(new_line_seq) > 0:
        # height is the number of non zero element in shape_encoding_seq
        height_decoded = np.sum(new_line_seq)

        # width is the largest interval between two consecutive non zero elements in shape_encoding_seq
        new_line_indices = np.where(new_line_seq)[0]
        new_line_indices = np.insert(new_line_indices, 0, 0)
        diffs = np.diff(new_line_indices)
        width_decoded = max(1, np.max(diffs))
    else:
        # no effective new line token, so we assume the height~width ratio is 1:1
        height_decoded = int(math.sqrt(len(rgb_seq)))
        width_decoded = len(rgb_seq) // height_decoded

        # add positive new line token to new_line_seq, every width_decoded elements
        new_line_seq[width_decoded-1::width_decoded] = 1

        # in case of width_decoded * height_decoded < len(rgb_seq), truncate
        effective_seq_length = width_decoded * height_decoded
        rgb_seq = rgb_seq[:effective_seq_length]
        new_line_seq = new_line_seq[:effective_seq_length]
        mask_seq = mask_seq[:effective_seq_length]

    width_decoded += 1 # don't know why, but fix bug
    
    segment = np.zeros((height_decoded, width_decoded, 3))
    mask = np.zeros((height_decoded, width_decoded))

    # split segment_data into parts according to new_line_seq=True positions (splited parts could be in different length)
    split_indices = np.where(new_line_seq)[0] + 1 
    splited_rgb_lines = np.split(rgb_seq, split_indices)
    splited_mask_lines = np.split(mask_seq, split_indices)
    splited_rgb_lines = [x for x in splited_rgb_lines if len(x) > 0]
    splited_mask_lines = [x for x in splited_mask_lines if len(x) > 0]

    for row_id in range(len(splited_rgb_lines)):
        rgb_line = splited_rgb_lines[row_id]
        mask_line = splited_mask_lines[row_id]
        segment[row_id, :len(rgb_line), :] = rgb_line
        mask[row_id, :len(mask_line)] = mask_line
    
    # apply mask to the segment: set all masked pixels to 255
    segment[mask == 0] = 255

    segment = segment[:, :-1, :]
    mask = mask[:, :-1]
    segment = segment.astype(np.uint8)

    return segment, mask


def pad_to_square(img):
    # img: np array of h, w, 3
    h, w = img.shape[:2]
    if h > w:
        pad = (h - w) // 2
        img = np.pad(img, ((0, 0), (pad, pad), (0, 0)), mode='constant', constant_values=255)
    elif w > h:
        pad = (w - h) // 2
        img = np.pad(img, ((pad, pad), (0, 0), (0, 0)), mode='constant', constant_values=255)
    return img


def visualize_segments(sample_info, original_segment, reconstructed_segment):

    original_segment = pad_to_square(original_segment)
    reconstructed_segment = pad_to_square(reconstructed_segment)

    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(20, 5)
    # fig.suptitle(sample_info['name'])

    # ax[0].set_title('Original Image')
    ax[0].imshow(Image.open(sample_info['image_path']), aspect='auto')
    x, y, w, h = sample_info['bbox']
    rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=4)
    ax[0].add_patch(rect)
    ax[0].set_axis_off()

    # ax[1].set_title('Original Segment')
    ax[1].imshow(original_segment)
    ax[1].set_axis_off()

    # ax[2].set_title('Reconstructed Segment')
    ax[2].imshow(reconstructed_segment)
    ax[2].set_axis_off()

    return fig


def get_datasets(model, expand_mask_ratio=0, datasets=['sa1b', 'coco', 'lvis', 'v3det', 'visual_genome']):
    all_datasets = []
    if 'sa1b' in datasets:
        sa1b_dataset = SeqMaskDataset(
            dataset=SA1BDataset(sa1b_root='/home/dchenbs/workspace/datasets/sa1b'), 
            num_queries=model.config.num_queries, data_seq_length=model.config.data_seq_length,
            expand_mask_ratio=expand_mask_ratio,
        )
        all_datasets.append(sa1b_dataset)

    if 'coco' in datasets:
        coco_dataset = SeqMaskDataset(
            dataset=COCODataset(coco_root='/home/dchenbs/workspace/datasets/coco2017', split='val'), 
            num_queries=model.config.num_queries, data_seq_length=model.config.data_seq_length,
            text_features='data/text_features/coco_clip_rn50.npy',
            expand_mask_ratio=expand_mask_ratio,
        )
        all_datasets.append(coco_dataset)
    
    if 'lvis' in datasets:
        lvis_dataset = SeqMaskDataset(
            dataset=LVISDataset(lvis_root='/home/dchenbs/workspace/datasets/lvis', coco_root='/home/dchenbs/workspace/datasets/coco2017', split='val'), 
            num_queries=model.config.num_queries, data_seq_length=model.config.data_seq_length,
            text_features='data/text_features/lvis_clip_rn50.npy',
            expand_mask_ratio=expand_mask_ratio,
        )
        all_datasets.append(lvis_dataset)

    if 'v3det' in datasets:
        v3det_dataset = SeqMaskDataset(
            dataset=V3DetDataset(v3det_root='/home/dchenbs/workspace/datasets/v3det', split='val'), 
            num_queries=model.config.num_queries, data_seq_length=model.config.data_seq_length,
            text_features='data/text_features/v3det_clip_rn50.npy',
            expand_mask_ratio=expand_mask_ratio,
        )
        all_datasets.append(v3det_dataset)

    if 'visual_genome' in datasets:
        visual_genome_dataset = SeqMaskDataset(
            dataset=VisualGenomeDataset(visual_genome_root='/home/dchenbs/workspace/datasets/VisualGenome', split='val'), 
            num_queries=model.config.num_queries, data_seq_length=model.config.data_seq_length,
            text_features='data/text_features/visual_genome_clip_rn50.npy',
            expand_mask_ratio=expand_mask_ratio,
    )
        all_datasets.append(visual_genome_dataset)

    return all_datasets



@torch.no_grad()
def reconstruction_evaluation(model, datasets, num_steps=1, batch_size=1, num_visualizations=50):

    reconstruction_evaluation_results = {}
    all_visualizations = {}
    for dataset in datasets:
        print(f'Generating reconstructions for {dataset.dataset.dataset_name} dataset')
        overall_pixel_rmse = 0
        overall_aspect_ratio_rmse = 0
        overall_mask_rmse = 0
        visualizations = []
        for step in tqdm.tqdm(range(num_steps)):
            batch_data = []
            batch_sample_info = []
            for i in range(batch_size):
                this_data, this_sample_info = dataset[np.random.randint(0, len(dataset))]
                batch_data.append(this_data)
                batch_sample_info.append(this_sample_info)

            batch_data = torch.stack(batch_data).half().cuda()
            batch_latents = model.encode(batch_data)
            batch_reconstructed = model.generate(batch_latents, show_progress_bar=False)

            for i in range(batch_size):
                seq = batch_data[i]
                reconstructed = batch_reconstructed[i]
                sample_info = batch_sample_info[i]

                original_segment, original_mask = decode_image_from_seq(seq.float().cpu().numpy())
                reconstructed_segment, reconstructed_mask = decode_image_from_seq(reconstructed.float().cpu().numpy())

                original_height, original_width = original_segment.shape[:2]
                reconstructed_height, reconstructed_width = reconstructed_segment.shape[:2]
                min_width = min(original_width, reconstructed_width)
                min_height = min(original_height, reconstructed_height)
                
                pixel_rmse = np.sqrt(
                    np.mean((original_segment[:min_height, :min_width, :] - reconstructed_segment[:min_height, :min_width, :])**2))
                overall_pixel_rmse += pixel_rmse

                mask_rmse = np.sqrt(np.mean((original_mask[:min_height, :min_width] - reconstructed_mask[:min_height, :min_width])**2))
                overall_mask_rmse += mask_rmse

                aspect_ratio_rmse = np.sqrt((original_width/original_height - reconstructed_width/reconstructed_height)**2)
                overall_aspect_ratio_rmse += aspect_ratio_rmse

                if len(visualizations) < num_visualizations:
                    fig = visualize_segments(sample_info, original_segment, reconstructed_segment)
                    visualizations.append(fig)

        all_visualizations[dataset.dataset.dataset_name] = visualizations

        num_samples = num_steps * batch_size
        reconstruction_evaluation_result = {
            'pixel_rmse': overall_pixel_rmse/num_samples if overall_pixel_rmse > 0 else 0,
            'mask_rmse': overall_mask_rmse/num_samples if overall_mask_rmse > 0 else 0,
            'aspect_ratio_rmse': overall_aspect_ratio_rmse/num_samples if overall_aspect_ratio_rmse > 0 else 0,
        }
        reconstruction_evaluation_result = {k: round(v, 5) for k, v in reconstruction_evaluation_result.items()}

        print(reconstruction_evaluation_result)
        reconstruction_evaluation_results[dataset.dataset.dataset_name] = reconstruction_evaluation_result


    return reconstruction_evaluation_results, all_visualizations


def tsne_visualize(latents, ids, title=''):
    tsne = TSNE(n_components=2, random_state=42)
    latents_tsne = tsne.fit_transform(latents)

    fig = plt.figure(figsize=(10, 10))
    plt.scatter(latents_tsne[:, 0], latents_tsne[:, 1], c=ids, cmap='tab20', s=10, alpha=0.5)
    plt.axis('off')
    plt.title(title)
    # plt.show()
    return fig


def umap_visualize(latents, ids, title='', do_clustering=False):
    latents_umap = umap.UMAP().fit_transform(latents)
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(latents_umap[:, 0], latents_umap[:, 1], c=ids, cmap='tab20', s=10, alpha=0.5)
    plt.axis('off')
    plt.title(title)
    # plt.show()
    return fig


def linear_classification_evaluation(latents, ids, val_split_ratio=0.5):
    train_latents, val_latents, train_ids, val_ids = train_test_split(latents, ids, test_size=val_split_ratio, random_state=42)
    clf = LogisticRegression(random_state=42, max_iter=3000).fit(train_latents, train_ids)
    predicted_ids = clf.predict(val_latents)
    return accuracy_score(val_ids, predicted_ids)


def knn_classification_evaluation(latents, ids, val_split_ratio=0.5, k=5):
    
    train_latents, val_latents, train_ids, val_ids = train_test_split(latents, ids, test_size=val_split_ratio, random_state=42)
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(train_latents)

    overall_similarity = 0
    for i in tqdm.tqdm(range(len(val_latents))):
        latent = val_latents[i]
        distances, indices = knn.kneighbors([latent])
        class_ids = train_ids[indices[0]]
        class_id = np.argmax(np.bincount(class_ids))
        overall_similarity += (class_id == val_ids[i])
    return overall_similarity/len(val_latents)



@torch.no_grad()
def representation_evaluation(model, datasets, truncation=30000):

    representation_evaluation_results = {}
    all_features = {}
    for dataset in datasets:
        all_latents = []
        all_names = []
        print(f'Generating latent vectors for {dataset.dataset.dataset_name} dataset')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=1, pin_memory=True, prefetch_factor=4)
            
        for batch_data, batch_sample_info in tqdm.tqdm(dataloader):
            batch_data = batch_data.half().cuda()
                
            with torch.no_grad():
                batch_latents = model.encode(batch_data).cpu().numpy()
            all_latents.append(batch_latents)
            all_names.extend(batch_sample_info['name'])
        
        all_latents = np.concatenate(all_latents, axis=0)[:truncation]
        all_names = np.array(all_names)[:truncation]
        all_features[dataset.dataset.dataset_name] = {
            'latents': all_latents,
            'names': all_names,
        }

        unique_names = list(set(all_names))
        name2id = {name: i for i, name in enumerate(unique_names)}
        ids = np.array([name2id[name] for name in all_names])

        linaer_acc = linear_classification_evaluation(all_latents, ids, val_split_ratio=1/3)
        # print(f'Linear classification accuracy: {linaer_acc}')

        knn_acc = knn_classification_evaluation(all_latents, ids, val_split_ratio=1/3, k=8)
        # print(f'KNN classification accuracy: {knn_acc}')

        representation_evaluation_results[dataset.dataset.dataset_name] = {
            'linear_classification_accuracy': linaer_acc,
            'knn_classification_accuracy': knn_acc,
        }

    return representation_evaluation_results, all_features


"""
conda activate seq2seq-ae
cd /home/dchenbs/workspace/Seq2Seq-AutoEncoder

CUDA_VISIBLE_DEVICES=4 python evaluation.py \
    --model_dir "runs/Nov28_20-50-04_host19-SA1B-[327MB-16queries-1024]-[lr1e-05-bs16x1step-8gpu]/checkpoints/checkpoint_ep1_step1950k" \
    --reconstruction-step 20 --reconstruction-batch-size 5 --reconstruction-num-visualization 100 \
    --representation-truncation 100

"""

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Process command line arguments.')

    parser.add_argument('--model_dir', type=str, required=True, help='Model directory')
    parser.add_argument('--reconstruction-step', type=int, default=20, help='Number of steps for reconstruction evaluation')
    parser.add_argument('--reconstruction-batch-size', type=int, default=50, help='Batch size for reconstruction evaluation')
    parser.add_argument('--reconstruction-num-visualization', type=int, default=50, help='Number of samples for reconstruction visualization')

    parser.add_argument('--representation-truncation', type=int, default=30000, help='Number of samples for representation evaluation')

    args = parser.parse_args()
    model_dir = args.model_dir

    print(f'Loading model from {model_dir}')

    model = Seq2SeqAutoEncoderModel.from_pretrained(model_dir).half().cuda().eval()
    
    # reconstruction evaluation
    datasets = get_datasets(model, datasets=['sa1b', 'coco', 'lvis', 'v3det', 'visual_genome'])
    num_steps = args.reconstruction_step
    batch_size = args.reconstruction_batch_size
    reconstruction_evaluation_results, reconstruction_visualizations = reconstruction_evaluation(
        model, 
        datasets, 
        num_steps=num_steps, 
        batch_size=batch_size, 
        num_visualizations=args.reconstruction_num_visualization
        )
    pprint.pprint(reconstruction_evaluation_results)
    file_name = f'reconstruction_evaluation_{num_steps*batch_size}samples'
    json.dump(reconstruction_evaluation_results, open(os.path.join(model_dir, f'{file_name}.json'), 'w'), indent=4)
    pd.json_normalize(reconstruction_evaluation_results, sep='-').to_csv(os.path.join(model_dir, f'{file_name}.csv'), index=False)
    print(f'>>> Saved Reconstruction Evaluation Results to {model_dir}/{file_name}.json/csv')

    os.makedirs(os.path.join(model_dir, 'reconstruction_visualizations'), exist_ok=True)
    for dataset_name, visualizations in reconstruction_visualizations.items():
        for i, fig in enumerate(visualizations):
            fig.savefig(os.path.join(model_dir, 'reconstruction_visualizations', f'{dataset_name}-{i}.png'), bbox_inches='tight', pad_inches=0)
            plt.close(fig)

    
    # # representation evaluation
    # datasets = get_datasets(model, datasets=['coco', 'lvis', 'v3det'])
    # truncation = args.representation_truncation
    # representation_evaluation_results, all_features = representation_evaluation(
    #     model, 
    #     datasets, 
    #     truncation=truncation,
    #     )
    
    # pprint.pprint(representation_evaluation_results)
    # file_name = f'representation_evaluation_{truncation}samples'
    # json.dump(representation_evaluation_results, open(os.path.join(model_dir, f'{file_name}.json'), 'w'), indent=4)
    # pd.json_normalize(representation_evaluation_results, sep='-').to_csv(os.path.join(model_dir, f'{file_name}.csv'), index=False)
    # print(f'>>> Saved Representation Evaluation Results to {model_dir}/{file_name}.json/csv')

    # os.makedirs(os.path.join(model_dir, 'representation_visualizations'), exist_ok=True)
    # for dataset_name, features in all_features.items():
    #     latents = features['latents']
    #     ids = features['names']
    #     tsne_fig = tsne_visualize(latents, ids, title=f'TSNE-{dataset_name}')
    #     umap_fig = umap_visualize(latents, ids, title=f'UMAP-{dataset_name}')
    #     tsne_fig.savefig(os.path.join(model_dir, 'representation_visualizations', f'tsne-{dataset_name}.png'), bbox_inches='tight', pad_inches=0)
    #     umap_fig.savefig(os.path.join(model_dir, 'representation_visualizations', f'umap-{dataset_name}.png'), bbox_inches='tight', pad_inches=0)
    #     plt.close(tsne_fig)
    #     plt.close(umap_fig)

