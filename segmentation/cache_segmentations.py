import argparse
import json
import os
import pprint
import sys
import tqdm
import numpy as np
import pycocotools.mask as mask_util
import time
from torch.utils.tensorboard import SummaryWriter
import pickle
import random
random.seed(0)

sys.path.append('../data/')

from PIL import Image
from segmentation import Segmenter
from sharegpt4v import ShareGPT4V


def main(args):
    os.makedirs(f'{args.cache_root}/{args.dataset_identifier}', exist_ok=True)

    share_gpt4v_dataset = ShareGPT4V(os.path.join(args.sharegpt4v_path, f'{args.dataset_identifier}.json'), only_return_img_path=True)
    share_gpt4v_dataset.validate_exist()

    all_paths = []
    for img_path in share_gpt4v_dataset:
        all_paths.append(img_path)

    all_paths = all_paths[:args.debug_cutoff]

    all_paths = list(set(all_paths))
    print(f'Found {len(all_paths)} images in {args.dataset_identifier}')

    with open(args.segmenter_config, 'r') as f:
        config = json.load(f)
        pprint.pprint(config)
    segmenter = Segmenter(config['model_name'], config['checkpoint'], **config['kwargs'])

    cache = {}
    start_idx = args.start_parts * args.sample_per_part
    end_idx = args.end_parts * args.sample_per_part if args.end_parts != -1 else -1
    all_paths = all_paths[start_idx:end_idx]
    
    profile_dir = f'{args.cache_root}/{args.dataset_identifier}/profiling/[{args.start_parts}:{args.end_parts}]'
    os.system(f'rm -rf {profile_dir}')
    writer = SummaryWriter(profile_dir)
    global_step = 0

    for img_path in tqdm.tqdm(all_paths, desc=f'[Parts {args.start_parts}:{args.end_parts}; Range {start_idx}:{end_idx}]'):
        
        try:
            masks, profile = segmenter(img_path, profiling=True, resize_to_max_of=args.resize_to_max_of)
            writer.add_scalars(f'profile', profile, global_step=global_step)
            writer.add_scalar(f'sample/n masks', len(masks), global_step=global_step)
            writer.add_scalar(f'sample/image size', masks[0]['segmentation'].shape[0]*masks[0]['segmentation'].shape[1], global_step=global_step)
            global_step+=1
        except Exception as e:
            if e=='KeyboardInterrupt':
                raise e
            else:
                print(f'Failed to segment {img_path}')
                continue

        for mask in masks:
            mask['segmentation'] = mask_util.encode(np.asfortranarray(mask['segmentation']))
            mask['segmentation']['counts'] = mask['segmentation']['counts'].decode('utf-8')
            mask['area'] = int(mask['area'])
        cache[img_path] = masks

        if len(cache) >= args.sample_per_part:
            # json.dump(cache, open(f'{args.cache_root}/{args.dataset_identifier}/part{args.start_parts}.json', 'w'))
            with open(f'{args.cache_root}/{args.dataset_identifier}/part{args.start_parts}.pkl', 'wb') as f:
                pickle.dump(cache, f)
            print(f'part{args.start_parts} saved')

            cache = {}
            args.start_parts += 1

    if len(cache) > 0:
        # json.dump(cache, open(f'{args.cache_root}/{args.dataset_identifier}/part{args.start_parts}.json', 'w'), indent=2)
        with open(f'{args.cache_root}/{args.dataset_identifier}/part{args.start_parts}.pkl', 'wb') as f:
            pickle.dump(cache, f)
        print(f'part{args.start_parts} saved')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_identifier', required=True) 
        # sharegpt4v_instruct_gpt4-vision_cap100k
        # share-captioner_coco_lcs_sam_1246k_1107
        # sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k
    parser.add_argument('--sharegpt4v_path', required=True)
    parser.add_argument('--segmenter_config', required=True)
    parser.add_argument('--start_parts', type=int, default=0)
    parser.add_argument('--end_parts', type=int, default=0)
    parser.add_argument('--sample_per_part', type=int, default=10)
    parser.add_argument('--debug_cutoff', type=int, default=-1)
    parser.add_argument('--cache_root', default='cached_segments')
    parser.add_argument('--resize_to_max_of', type=int, default=-1)
    args = parser.parse_args()

    main(args)


"""
cd /home/dchenbs/workspace/Seq2Seq-AutoEncoder/segmentation
conda activate seq2seq-ae
CUDA_VISIBLE_DEVICES=0 python cache_segmentations.py \
    --resize_to_max_of 1024 \
    --start_parts 0 --end_parts -1 --sample_per_part 10000 \
    --dataset_identifier sharegpt4v_instruct_gpt4-vision_cap100k \
    --sharegpt4v_path /home/dchenbs/workspace/datasets/sharegpt4v/ShareGPT4V \
    --segmenter_config /home/dchenbs/workspace/Seq2Seq-AutoEncoder/segmentation/mobile_sam_v2_l2.json \
    --cache_root /home/dchenbs/workspace/Seq2Seq-AutoEncoder/segmentation/cached_segments
"""

"""
cd /home/dchenbs/workspace/Seq2Seq-AutoEncoder/segmentation
conda activate seq2seq-ae
CUDA_VISIBLE_DEVICES=3 python cache_segmentations.py \
    --resize_to_max_of 1024 \
    --start_parts 18 --end_parts -1 --sample_per_part 10000 \
    --dataset_identifier sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k \
    --sharegpt4v_path /home/dchenbs/workspace/datasets/sharegpt4v/ShareGPT4V \
    --segmenter_config /home/dchenbs/workspace/Seq2Seq-AutoEncoder/segmentation/mobile_sam_v2_l2.json \
    --cache_root /home/dchenbs/workspace/Seq2Seq-AutoEncoder/segmentation/cached_segments
"""

"""
cd /home/dchenbs/workspace/Seq2Seq-AutoEncoder/segmentation
conda activate seq2seq-ae
CUDA_VISIBLE_DEVICES=0 python cache_segmentations.py \
    --resize_to_max_of 1024 \
    --start_parts 0 --end_parts -1 --sample_per_part 10000 \
    --dataset_identifier share-captioner_coco_lcs_sam_1246k_1107 \
    --sharegpt4v_path /home/dchenbs/workspace/datasets/sharegpt4v/ShareGPT4V \
    --segmenter_config /home/dchenbs/workspace/Seq2Seq-AutoEncoder/segmentation/mobile_sam_v2_l2.json \
    --cache_root /home/dchenbs/workspace/Seq2Seq-AutoEncoder/segmentation/cached_segments
"""