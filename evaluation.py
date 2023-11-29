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

np.random.seed(42)
random.seed(42)

from data.dataset import get_dataset, SeqMaskDataset, LVISDataset, V3DetDataset, COCODataset, VisualGenomeDataset, SA1BDataset
from model import Seq2SeqAutoEncoderModel, Seq2SeqAutoEncoderConfig

def decode_image_from_seq(seq):
    seq = seq.numpy()
    segment_data = seq[1:, :3]
    shape_encoding_seq = seq[1:, 3]
    is_data_seq = seq[1:, 4]

    shape_encoding_seq = shape_encoding_seq > 0.3
    is_data_seq = is_data_seq > 0.5

    # find the last positive element in shape_encoding_seq, and use it to truncate data and sequences
    non_zero_indices = np.nonzero(shape_encoding_seq)[0]
    if non_zero_indices.size > 0:
        last_positive_index = non_zero_indices[-1]
    else:
        last_positive_index = len(shape_encoding_seq) - 1
    
    shape_encoding_seq = shape_encoding_seq[:last_positive_index+1]
    is_data_seq = is_data_seq[:last_positive_index+1]
    segment_data = segment_data[:last_positive_index+1] * 255

    # height is the number of non zero element in shape_encoding_seq
    height_decoded = np.sum(shape_encoding_seq)

    # width is the largest interval between two consecutive non zero elements in shape_encoding_seq
    width_decoded = 0
    true_indices = np.where(shape_encoding_seq)[0]
    true_indices = np.insert(true_indices, 0, 0)
    diffs = np.diff(true_indices)
    if diffs.size > 0:
        width_decoded = np.max(diffs)

    width_decoded += 1 # don't know why, but fix bug

    segment = np.zeros((height_decoded, width_decoded, 3))
    mask = np.zeros((height_decoded, width_decoded))

    # split segment_data into parts according to shape_encoding_seq=True positions, splited parts could be in different length
    split_indices = np.where(shape_encoding_seq)[0]
    split_indices += 1
    split_segment_data = np.split(segment_data, split_indices)
    split_segment_data = [x for x in split_segment_data if len(x) > 0]

    split_is_data = np.split(is_data_seq, split_indices)
    split_is_data = [x for x in split_is_data if len(x) > 0]

    for row_id in range(len(split_segment_data)):
        segment_split = split_segment_data[row_id]
        mask_split = split_is_data[row_id]
        segment[row_id, :len(segment_split), :] = segment_split
        mask[row_id, :len(mask_split)] = mask_split
    
    # apply mask to the segment: set all masked pixels to 255
    segment[mask == 0] = 255  

    if width_decoded > 1:
        segment = segment[:, :-1, :]
    segment = transforms.ToPILImage()(segment.astype(np.uint8))

    return segment, mask


def visualize_segments(sample_info, original_segment, reconstructed_segment, original_mask, reconstructed_mask):

    fig, ax = plt.subplots(1, 5)
    fig.set_size_inches(25, 5)
    fig.suptitle(sample_info['name'])

    ax[0].set_title('Original Image')
    ax[0].imshow(Image.open(sample_info['image_path']))
    x, y, w, h = sample_info['bbox']
    rect = plt.Rectangle((x, y), w, h, fill=False, color='red')
    ax[0].add_patch(rect)

    ax[1].set_title('Original Segment')
    ax[1].imshow(original_segment)

    ax[2].set_title('Reconstructed Segment')
    ax[2].imshow(reconstructed_segment)

    ax[3].set_title('Original Mask')
    ax[3].imshow(original_mask)

    ax[4].set_title('Reconstructed Mask')
    ax[4].imshow(reconstructed_mask)

    return fig

    
from sklearn.manifold import TSNE
def tsne_visualize(latents, ids):
    tsne = TSNE(n_components=2, random_state=42)
    latents_tsne = tsne.fit_transform(latents)

    fig = plt.figure(figsize=(10, 10))
    plt.scatter(latents_tsne[:, 0], latents_tsne[:, 1], c=ids, cmap='tab20', s=10, alpha=0.5)
    plt.axis('off')
    plt.title(f'T-SNE of {len(ids)} samples')
    # plt.savefig(os.path.join(vis_dir, f'tsne-{dataset.dataset.dataset_name}-{model_dir.split("/")[-1]}.png'), bbox_inches='tight', pad_inches=0)
    plt.show()


from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
def get_knn_similarity(latents, references, k=10):
    if None in references:
        return 0
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(latents)
    overall_similarity = 0
    
    for i in tqdm.tqdm(range(len(latents))):
        latent = latents[i]
        distances, indices = knn.kneighbors([latent])
        similarities = cosine_similarity(references[indices[0]], references[i].reshape(1, -1))
        overall_similarity += np.mean(similarities)
    
    return overall_similarity/len(latents)


from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_linear_projection_cosine_similarity(latents, references, train_val_split_ratio=0.5):
    if latents is None or references is None or len(latents) != len(references):
        return 0
    
    train_val_split_index = int(len(latents) * train_val_split_ratio)

    model = LinearRegression()
    model.fit(latents[:train_val_split_index], references[:train_val_split_index])

    predicted_references = model.predict(latents[train_val_split_index:])
    ground_truth_references = references[train_val_split_index:]

    similarity = [cosine_similarity(predicted_references[i].reshape(1, -1), ground_truth_references[i].reshape(1, -1)) for i in range(len(latents)-train_val_split_index)]

    return np.mean(similarity), (predicted_references, ground_truth_references)


def get_datasets(model, expand_mask_ratio=0):
    sa1b_dataset = SeqMaskDataset(
        dataset=SA1BDataset(sa1b_root='/home/dchenbs/workspace/datasets/sa1b'), 
        num_queries=model.config.num_queries, data_seq_length=model.config.data_seq_length,
        expand_mask_ratio=expand_mask_ratio,
    )

    coco_dataset = SeqMaskDataset(
        dataset=COCODataset(coco_root='/home/dchenbs/workspace/datasets/coco2017', split='val'), 
        num_queries=model.config.num_queries, data_seq_length=model.config.data_seq_length,
        text_features='data/text_features/coco_clip_rn50.npy',
        expand_mask_ratio=expand_mask_ratio,
    )

    lvis_dataset = SeqMaskDataset(
        dataset=LVISDataset(lvis_root='/home/dchenbs/workspace/datasets/lvis', coco_root='/home/dchenbs/workspace/datasets/coco2017', split='val'), 
        num_queries=model.config.num_queries, data_seq_length=model.config.data_seq_length,
        text_features='data/text_features/lvis_clip_rn50.npy',
        expand_mask_ratio=expand_mask_ratio,
    )

    v3det_dataset = SeqMaskDataset(
        dataset=V3DetDataset(v3det_root='/home/dchenbs/workspace/datasets/v3det', split='val'), 
        num_queries=model.config.num_queries, data_seq_length=model.config.data_seq_length,
        text_features='data/text_features/v3det_clip_rn50.npy',
        expand_mask_ratio=expand_mask_ratio,
    )

    visual_genome_dataset = SeqMaskDataset(
        dataset=VisualGenomeDataset(visual_genome_root='/home/dchenbs/workspace/datasets/VisualGenome', split='val'), 
        num_queries=model.config.num_queries, data_seq_length=model.config.data_seq_length,
        text_features='data/text_features/visual_genome_clip_rn50.npy',
        expand_mask_ratio=expand_mask_ratio,
    )

    return [coco_dataset, lvis_dataset, v3det_dataset, visual_genome_dataset, sa1b_dataset]



def linear_evaluation(model, datasets, num_steps=2, batch_size=50, visualize=False):
    correlation_evaluation_results = {}
    for dataset in datasets:
        all_latents = []
        all_sample_info = []
        print(f'Generating latent vectors for {dataset.dataset.dataset_name} dataset')
        for step in tqdm.tqdm(range(num_steps)):
            batch_data = []
            for i in range(batch_size):
                this_data, this_sample_info = dataset[np.random.randint(0, len(dataset))]
                batch_data.append(this_data)
                this_sample_info['class_id'] = dataset.dataset.class_name_to_class_id(this_sample_info['name'])
                all_sample_info.append(this_sample_info)
            batch_data = torch.stack(batch_data).cuda()

            with torch.no_grad():
                batch_latents = model.encode(batch_data).cpu().numpy()
            all_latents.append(batch_latents)
        
        all_latents = np.concatenate(all_latents, axis=0)
        all_ids = np.array([x['class_id'] for x in all_sample_info])
        all_text_features = np.array([x['text_feature'] for x in all_sample_info])

        print('Calculating linear projection cosine similarity')
        linear_similarity, (prediction, ground_truth) = get_linear_projection_cosine_similarity(all_latents, all_text_features)
        print(f'Linear projection cosine similarity: {linear_similarity:.4f}')

        correlation_evaluation_results[dataset.dataset.dataset_name] = {
            'cosine_similarity': float(linear_similarity),
        }
        if visualize:
            tsne_visualize(all_latents, all_ids)
            tsne_visualize(np.concatenate([prediction, ground_truth], axis=0), np.concatenate([np.ones(len(prediction)), np.zeros(len(ground_truth))], axis=0))
    return correlation_evaluation_results

@torch.no_grad()
def reconstruction_evaluation(model, datasets, num_steps=1, batch_size=1, visualize=False):

    reconstruction_evaluation_results = {}
    for dataset in datasets:
        print(f'Generating reconstructions for {dataset.dataset.dataset_name} dataset')
        overall_pixel_rmse = 0
        overall_aspect_ratio_rmse = 0
        overall_mask_rmse = 0
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

                original_segment, original_mask = decode_image_from_seq(seq.float().cpu())
                reconstructed_segment, reconstructed_mask = decode_image_from_seq(reconstructed.float().cpu())

                original_width, original_height = original_segment.size
                reconstructed_width, reconstructed_height = reconstructed_segment.size
                min_width = min(original_width, reconstructed_width)
                min_height = min(original_height, reconstructed_height)
                
                pixel_rmse = np.sqrt(
                    np.mean((np.array(original_segment) - np.array(reconstructed_segment.resize((original_width, original_height))))**2))
                overall_pixel_rmse += pixel_rmse

                mask_rmse = np.sqrt(np.mean((original_mask[:min_height, :min_width] - reconstructed_mask[:min_height, :min_width])**2))
                overall_mask_rmse += mask_rmse

                aspect_ratio_rmse = np.sqrt((original_width/original_height - reconstructed_width/reconstructed_height)**2)
                overall_aspect_ratio_rmse += aspect_ratio_rmse

                # print(f"Pixel RMSE: {pixel_rmse:.4f}, Mask RMSE: {mask_rmse:.4f}, Aspect Ratio RMSE: {aspect_ratio_rmse:.4f}")
                if visualize:
                    print(f"[{dataset.dataset.dataset_name}]: {sample_info['caption']}")
                    fig = visualize_segments(sample_info, original_segment, reconstructed_segment, original_mask, reconstructed_mask)
                    plt.show()
                    plt.close(fig)

        num_samples = num_steps * batch_size
        reconstruction_evaluation_result = {
            'pixel_rmse': overall_pixel_rmse/num_samples if overall_pixel_rmse > 0 else 0,
            'mask_rmse': overall_mask_rmse/num_samples if overall_mask_rmse > 0 else 0,
            'aspect_ratio_rmse': overall_aspect_ratio_rmse/num_samples if overall_aspect_ratio_rmse > 0 else 0,
        }
        reconstruction_evaluation_result = {k: round(v, 5) for k, v in reconstruction_evaluation_result.items()}

        print(reconstruction_evaluation_result)
        reconstruction_evaluation_results[dataset.dataset.dataset_name] = reconstruction_evaluation_result

    return reconstruction_evaluation_results


"""
python evaluation.py --cuda_visible_devices "4" --steps 50 150 250 350 450 550 650 750 850 950
python evaluation.py --cuda_visible_devices "5" --steps 100 200 300 400 500 600 700 800 900 1000
"""
if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Process command line arguments.')

    parser.add_argument('--cuda_visible_devices', type=str, required=True, help='CUDA_VISIBLE_DEVICES value')
    parser.add_argument('--steps', nargs='+', type=int, required=True, help='Steps for evaluation')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    steps = args.steps
    model_dir_templete ='/home/dchenbs/workspace/Seq2Seq-AutoEncoder/runs/Nov14_17-31-06_host19-SA1B-[327MB-16queries-1024]-[lr1e-05-bs16x1step-8gpu]/checkpoints/checkpoint_ep0_step1000k'

    for step in args.steps:
        model_dir = model_dir_templete.replace('1000k', f'{step}k')
        print(f'Loading model from {model_dir}')

        model = Seq2SeqAutoEncoderModel.from_pretrained(model_dir).half().cuda().eval()
        datasets = get_datasets(model)

        # num_steps = 100
        # batch_size = 100
        # linear_evaluation_results = linear_evaluation(
        #     model, 
        #     datasets, 
        #     num_steps=num_steps, 
        #     batch_size=batch_size, 
        #     visualize=True
        #     )
        # pprint.pprint(linear_evaluation_results)
        # file_name = f'linear_evaluation_{num_steps*batch_size}samples'
        # json.dump(linear_evaluation_results, open(os.path.join(model_dir, f'{file_name}.json'), 'w'), indent=4)
        # pd.json_normalize(linear_evaluation_results, sep='-').to_csv(os.path.join(model_dir, f'{file_name}.csv'), index=False)
        # print(f'>>> Saved KNN Evluation Results to {model_dir}/{file_name}.json/csv')


        num_steps = 20
        batch_size = 50
        reconstruction_evaluation_results = reconstruction_evaluation(
            model, 
            datasets, 
            num_steps=num_steps, 
            batch_size=batch_size, 
            visualize=False
            )
        pprint.pprint(reconstruction_evaluation_results)
        file_name = f'reconstruction_evaluation_{num_steps*batch_size}samples'
        json.dump(reconstruction_evaluation_results, open(os.path.join(model_dir, f'{file_name}.json'), 'w'), indent=4)
        pd.json_normalize(reconstruction_evaluation_results, sep='-').to_csv(os.path.join(model_dir, f'{file_name}.csv'), index=False)
        print(f'>>> Saved Reconstruction Evaluation Results to {model_dir}/{file_name}.json/csv')