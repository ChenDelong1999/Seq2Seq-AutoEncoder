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

np.random.seed(42)
random.seed(42)

from model import Seq2SeqAutoEncoderModel, Seq2SeqAutoEncoderConfig
from evaluation import decode_image_from_seq, visualize_segments, tsne_visualize, get_knn_similarity, get_datasets, linear_evaluation, reconstruction_evaluation


from data.dataset import get_dataset, SeqMaskDataset, LVISDataset, V3DetDataset, COCODataset, VisualGenomeDataset, SA1BDataset

def get_datasets(model, expand_mask_ratio=0):

    coco_dataset = SeqMaskDataset(
        dataset=COCODataset(coco_root='/home/dchenbs/workspace/datasets/coco2017', split='val'), 
        num_queries=model.config.num_queries, data_seq_length=model.config.data_seq_length,
        # text_features='data/text_features/coco_clip_rn50.npy',
        expand_mask_ratio=expand_mask_ratio,
    )

    lvis_dataset = SeqMaskDataset(
        dataset=LVISDataset(lvis_root='/home/dchenbs/workspace/datasets/lvis', coco_root='/home/dchenbs/workspace/datasets/coco2017', split='val'), 
        num_queries=model.config.num_queries, data_seq_length=model.config.data_seq_length,
        # text_features='data/text_features/lvis_clip_rn50.npy',
        expand_mask_ratio=expand_mask_ratio,
    )

    v3det_dataset = SeqMaskDataset(
        dataset=V3DetDataset(v3det_root='/home/dchenbs/workspace/datasets/v3det', split='val'), 
        num_queries=model.config.num_queries, data_seq_length=model.config.data_seq_length,
        # text_features='data/text_features/v3det_clip_rn50.npy',
        expand_mask_ratio=expand_mask_ratio,
    )

    return [v3det_dataset, coco_dataset, lvis_dataset, ]


if __name__ == '__main__':

    model_configs = [
        # 'runs/Nov14_17-31-06_host19-SA1B-[327MB-16queries-1024]-[lr1e-05-bs16x1step-8gpu]/checkpoints/checkpoint_ep0_step50k/config.json',
        # 'runs/Nov14_17-31-06_host19-SA1B-[327MB-16queries-1024]-[lr1e-05-bs16x1step-8gpu]/checkpoints/checkpoint_ep0_step200k',
        # 'runs/Nov14_17-31-06_host19-SA1B-[327MB-16queries-1024]-[lr1e-05-bs16x1step-8gpu]/checkpoints/checkpoint_ep0_step400k',
        # 'runs/Nov14_17-31-06_host19-SA1B-[327MB-16queries-1024]-[lr1e-05-bs16x1step-8gpu]/checkpoints/checkpoint_ep0_step600k',
        # 'runs/Nov14_17-31-06_host19-SA1B-[327MB-16queries-1024]-[lr1e-05-bs16x1step-8gpu]/checkpoints/checkpoint_ep0_step800k',
        # 'runs/Nov14_17-31-06_host19-SA1B-[327MB-16queries-1024]-[lr1e-05-bs16x1step-8gpu]/checkpoints/checkpoint_ep0_step1000k',

        'runs/Nov28_20-50-04_host19-SA1B-[327MB-16queries-1024]-[lr1e-05-bs16x1step-8gpu]/checkpoints/checkpoint_ep0_step200k',
        'runs/Nov28_20-50-04_host19-SA1B-[327MB-16queries-1024]-[lr1e-05-bs16x1step-8gpu]/checkpoints/checkpoint_ep0_step400k',
        'runs/Nov28_20-50-04_host19-SA1B-[327MB-16queries-1024]-[lr1e-05-bs16x1step-8gpu]/checkpoints/checkpoint_ep0_step600k',
        'runs/Nov28_20-50-04_host19-SA1B-[327MB-16queries-1024]-[lr1e-05-bs16x1step-8gpu]/checkpoints/checkpoint_ep0_step800k',
        'runs/Nov28_20-50-04_host19-SA1B-[327MB-16queries-1024]-[lr1e-05-bs16x1step-8gpu]/checkpoints/checkpoint_ep0_step1000k',
        ]
    
    feature_cache_dir = '/home/dchenbs/workspace/cache/SeqAE_features/'
    print(model_configs)

    for model_config in model_configs:
        if model_config.endswith('.json'):
            config = Seq2SeqAutoEncoderConfig.from_json_file(model_config)
            model = Seq2SeqAutoEncoderModel(config)
        else:
            model = Seq2SeqAutoEncoderModel.from_pretrained(model_config)
        model = model.half().cuda().eval()
        print(f'>>> Loaded model from {model_config}')

        datasets = get_datasets(model, expand_mask_ratio=0)
        for dataset in datasets:
            all_latents = []
            all_names = []

            print(f'Generating latent vectors for {dataset.dataset.dataset_name} dataset')

            dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=8, pin_memory=True)
            for batch_data, batch_sample_info in tqdm.tqdm(dataloader):
                batch_data = batch_data.half().cuda()
                all_names.extend(batch_sample_info['name'])
                    
                with torch.no_grad():
                    batch_latents = model.encode(batch_data).cpu().numpy()
                all_latents.append(batch_latents)
            
            all_latents = np.concatenate(all_latents, axis=0)
            all_names = np.array(all_names)

            print(f'all_latents: {all_latents.shape}\n{all_latents}')
            print(f'all_names: {all_names.shape}\n{all_names}')

            features = {
                'latents': all_latents,
                'names': all_names,
            }
            if '.json' in model_config:
                feature_file = f'{feature_cache_dir}/random-{dataset.dataset.dataset_name}.npy'
            else:
                feature_file = f'{feature_cache_dir}/{model_config.split("/")[1][:5]}-{model_config.split("/")[-1]}-{dataset.dataset.dataset_name}.npy'
            np.save(feature_file, features)