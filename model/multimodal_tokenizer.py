import torch
import torchvision
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import tqdm
import datasets
from utils import get_params_count_summary
import json
import time
import copy
import math
import os

import datasets
from data.sharegpt4v import ShareGPT4V

from transformers import CodeGenTokenizerFast
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import pycocotools.mask as mask_util
import pickle




class MultimodalTokenizer(CodeGenTokenizerFast):

    def __init__(self, segmenter, seqae_config, max_seg_per_img=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.segmenter = segmenter
        self.seqae_config = seqae_config
        self.max_seg_per_img = max_seg_per_img

        self.start_token = "<|startofimage|>"
        self.end_token = "<|endofimage|>"
        self.seg_token = "<|seg|>"
        self.add_tokens([self.start_token, self.end_token, self.seg_token])
        self.pad_token = self.eos_token
        self.cache = {}

    def load_cached_segments(self, cached_segments):
        if cached_segments.endswith('.pkl'):
            with open(cached_segments, 'rb') as f:
                cache = pickle.load(f)
        else:
            cache_files = os.listdir(cached_segments)
            cache = {}
            for cache_file in cache_files:
                if cache_file.endswith('.pkl'):
                    with open(os.path.join(cached_segments, cache_file), 'rb') as f:
                        this_cache = pickle.load(f)
                        cache.update(this_cache)
                        print(f'added {len(this_cache)} cached images from {cache_file}, total {len(cache)} cached images now.')

        for k, v in cache.items():
            if k not in self.cache:
                self.cache[k] = v

        print(f'Loaded {len(cache)} cached images from {cached_segments}, total {len(self.cache)} cached images now.')

    def get_sequences_and_bboxes(self, masks, image):

        def resize(segment):
            h = segment['patch'].size(1)
            w = segment['patch'].size(2)
            if h * w > (self.seqae_config.data_seq_length):
                ratio_to_maxlength = np.sqrt(self.seqae_config.data_seq_length / (h * w))
                h = int(h * ratio_to_maxlength)
                w = int(w * ratio_to_maxlength)
                segment['patch'] = torchvision.transforms.Resize([h, w], antialias=True)(segment['patch'])
                segment['segmentation'] = torchvision.transforms.Resize([h, w], antialias=True)(segment['segmentation'][None, :, :])[0]

            return segment
        
        def encode_to_sequence(segment):           
            # the following function is used during traning SeqAE. Now there is a faster tensor based implementation (encode_to_sequence)
            # so this one is deprecated, but I keep it here for reference 
            """
            def encode_to_sequence_slow(segment):
                # segment['patch'] is torch tensor with shape (C, H, W)
                h, w = segment['patch'].shape[1:]
                sequence = []
                for i in range(h):
                    for j in range(w):
                        pixel_data = segment['patch'][:, i, j]# / 255.0
                        is_rightmost = 1 if j == w - 1 else 0
                        is_non_masked = int(segment['segmentation'][i, j])
                        sequence.append(pixel_data.tolist() + [is_rightmost, is_non_masked])
                sequence = np.array(sequence) 

                # pad the sequence to max_seq_length with zeros
                if len(sequence) < self.seqae_config.data_seq_length:
                    sequence = np.concatenate((sequence, np.zeros((self.seqae_config.data_seq_length - len(sequence), self.seqae_config.input_channels))))

                # add the query place holder to the end of the sequence
                sequence = np.concatenate((sequence, np.zeros((self.seqae_config.num_queries, self.seqae_config.input_channels))))
                # add one all zero column to the start 
                sequence = np.concatenate((np.zeros((1, sequence.shape[1])), sequence), axis=0)

                return torch.from_numpy(sequence)
            """  
            
            # segment['patch'] is torch tensor with shape (C, H, W)
            patch = segment['patch']# / 255.0
            h, w = patch.shape[1:]

            # Create is_rightmost and is_non_masked tensors
            is_rightmost = torch.zeros((h, w), dtype=torch.float32)
            is_rightmost[:, -1] = 1
            is_non_masked = segment['segmentation'].float()
            
            # stack them together with patch
            patch = patch.permute(1, 2, 0) # (H, W, C)
            patch = patch.reshape(-1, patch.shape[-1])  # (H*W, C)
            is_rightmost = is_rightmost.reshape(-1, 1) # (H*W, 1)
            is_non_masked = is_non_masked.reshape(-1, 1) # (H*W, 1)
            sequence = torch.cat([patch, is_rightmost, is_non_masked], dim=-1).view(-1, 5)

            # pad the sequence to max_seq_length with zeros
            if len(sequence) < self.seqae_config.data_seq_length:
                sequence = torch.cat((sequence, torch.zeros((self.seqae_config.data_seq_length - len(sequence), self.seqae_config.input_channels))))

            # add the query place holder to the end of the sequence
            sequence = torch.cat((sequence, torch.zeros((self.seqae_config.num_queries, self.seqae_config.input_channels))))

            # add one all zero column to the start 
            sequence = torch.cat((torch.zeros((1, sequence.shape[1])), sequence), dim=0)

            return sequence

        segment_sequences = []
        bboxes = []

        for mask in masks:
            segmentation = mask["segmentation"]

            bbox = mask["bbox"]
            bbox[2] = 1 if bbox[2] == 0 else bbox[2]
            bbox[3] = 1 if bbox[3] == 0 else bbox[3]
            x, y, w, h = bbox

            segment = {
                "patch": image[:, y:y+h, x:x+w],
                "segmentation": segmentation[y:y+h, x:x+w],
            }

            assert segment['patch'].size(1) == segment['segmentation'].size(0) and segment['patch'].size(2) == segment['segmentation'].size(1)

            segment = resize(segment)
            segment['patch'] = segment['patch'] * segment['segmentation'][None, :, :]
            
            segment_sequence = encode_to_sequence(segment)
            segment_sequences.append(segment_sequence)
            bboxes.append(bbox)

        segment_sequences = torch.stack(segment_sequences, dim=0)
        bboxes = torch.tensor(bboxes)
        return segment_sequences, bboxes
    
    def get_segmentation(self, image_path):
        if image_path in self.cache:
            masks = copy.deepcopy(self.cache[image_path])
            for mask in masks:
                mask['segmentation'] = mask_util.decode(mask['segmentation']).astype(bool)
        else:
            masks = self.segmenter(image_path)
        
        for mask in masks:
            if type(mask['segmentation']) == np.ndarray:
                mask['segmentation'] = torch.from_numpy(mask['segmentation'])
        
        return masks
    

    def __call__(self, texts, *args, **kwargs):
        img_path_start, img_path_end = "<img_path>", "</img_path>"
        all_sequences, all_bboxes, all_texts = [], [], []

        for text in texts:
            this_text_sequences, this_text_bboxes = [], []
            if img_path_start in text and img_path_end in text:
                while img_path_start in text and img_path_end in text:
                    start_index, end_index = text.index(img_path_start), text.index(img_path_end)
                    image_path = text[start_index + len(img_path_start):end_index]

                    masks = self.get_segmentation(image_path)
                    masks.sort(key=lambda x: x['area'], reverse=True)
                    masks = masks[:self.max_seg_per_img]

                    h, w = masks[0]['segmentation'].shape[:2]
                    image = torchvision.transforms.ToTensor()(Image.open(image_path).convert('RGB'))
                    image = torchvision.transforms.Resize([h, w], antialias=True)(image)

                    segment_sequences, bboxes = self.get_sequences_and_bboxes(masks, image)
                    this_text_sequences.append(segment_sequences)
                    this_text_bboxes.append(bboxes)

                    # Replace image path with segment tokens <|seg|>
                    text = text[:start_index] + self.start_token + self.seg_token*len(masks) + self.end_token + text[end_index + len(img_path_end):]

                this_text_sequences = torch.stack(this_text_sequences)
                this_text_bboxes = torch.stack(this_text_bboxes)

            all_texts.append(text)
            all_sequences.append(this_text_sequences)
            all_bboxes.append(this_text_bboxes)

        inputs = super().__call__(all_texts, *args, **kwargs)

        inputs['segment_sequences'] = all_sequences # a list of `batch_size` elements, each element is a tensor of [image_num, num_segments, seq_length, input_channels]
        inputs['bboxes'] = all_bboxes # a list of `batch_size` elements, each element is a tensor of [image_num, num_segments, 4] representing bounding boxes in x, y, w, h

        return inputs
    
    
    def decode_image(self, segment_sequences, bboxes, filling='pixel'):

        # segment_sequences: torch tensor [num_segments, seq_length, input_channels]
        # bboxes: torch tensor [num_segments, 4] (x, y, w, h)

        # first, get the image size by finding the largest x+w and y+h
        # rightmost = torch.max(bboxes[:, 0] + bboxes[:, 2])
        # bottommost = torch.max(bboxes[:, 1] + bboxes[:, 3])
        rightmost = torch.max(bboxes[:, 2])+1
        bottommost = torch.max(bboxes[:, 3])+1
        canvas = np.zeros((int(bottommost), int(rightmost), 3)).astype(np.uint8)

        for i in range(len(segment_sequences)):
            x, y, w, h = bboxes[i]
            segment, mask = self.decode_image_from_seq(segment_sequences[i].numpy())
            segment = cv2.resize(segment, (int(w), int(h)))
            mask = cv2.resize(mask, (int(w), int(h)))
            # canvas[y:y+h, x:x+w, :] = segment * mask[:, :, None]

            # only modify the canvas where mask is 1
            if filling=='average': # use the average color within the mask to fill the canavas segment region
                avg_color = np.mean(segment, axis=(0, 1))
                canvas[y:y+h, x:x+w, :] = canvas[y:y+h, x:x+w, :] * (1 - mask[:, :, None]) + avg_color * mask[:, :, None]
            elif filling=='random':
                random_color = np.random.randint(0, 256, (3,))
                canvas[y:y+h, x:x+w, :] = canvas[y:y+h, x:x+w, :] * (1 - mask[:, :, None]) + random_color * mask[:, :, None]
            elif filling=='pixel':
                canvas[y:y+h, x:x+w, :] = canvas[y:y+h, x:x+w, :] * (1 - mask[:, :, None]) + segment * mask[:, :, None]
        return canvas


    def decode_image_from_seq(self, data_seq, new_line_threshold=0.5, mask_threshold=0.5):
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
