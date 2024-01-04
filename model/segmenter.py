import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(0)
import os
from scipy.ndimage import label
import time
import torch
import tqdm
from PIL import Image



class Segmenter():

    def __init__(
            self, 
            sam_ckpt,
            points_per_side = 32,
            points_per_batch = 64,
            pred_iou_thresh = 0.88,
            stability_score_thresh = 0.95,
            stability_score_offset = 1.0,
            box_nms_thresh = 0.7,
            crop_n_layers = 0,
            crop_nms_thresh = 0.7,
            crop_overlap_ratio = 512 / 1500,
            crop_n_points_downscale_factor = 1,
            min_mask_region_area = 0,
            device = 'cuda',
            ):
        
        if 'mobile_sam' in sam_ckpt:
            from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
            print(f'Loading Mobile SAM model from {sam_ckpt}')
            sam = sam_model_registry["vit_t"](checkpoint=sam_ckpt).to(device).eval()

        elif 'repvit_sam' in sam_ckpt:
            from repvit_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
            print(f'Loading RepVIT SAM model from {sam_ckpt}')
            sam = sam_model_registry["repvit"](checkpoint=sam_ckpt).to(device).eval()

        else:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            model_type = sam_ckpt.split('/')[-1][4:9]
            print(f'Loading SAM model {model_type} from {sam_ckpt}')
            sam = sam_model_registry[model_type](checkpoint=sam_ckpt).to(device).eval()

        self.generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            stability_score_offset=stability_score_offset,
            box_nms_thresh=box_nms_thresh,
            crop_n_layers=crop_n_layers,
            crop_nms_thresh=crop_nms_thresh,
            crop_overlap_ratio=crop_overlap_ratio,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
            )
        
        
    def __call__(self, image):        
        masks = self.generator.generate(image)
        masks = self.post_processing_masks(masks, image)
        return masks

    def expand_mask_blur(self, mask, kernel):
        mask = mask.copy()
        mask['segmentation'] = mask['segmentation'].astype(np.uint8)
        blurred_mask = cv2.filter2D(mask['segmentation'],-1,kernel)
        expanded_mask = (blurred_mask > 0).astype(bool)
        return expanded_mask


    def post_processing_masks(self, masks, image):

        kernel_size = int(min(image.shape[:2]) * 0.015) // 2 * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))

        masked_area = None
        post_processed_masks = []
        for mask in masks:
            expanded_mask = self.expand_mask_blur(mask, kernel)
            post_processed_masks.append({
                'mask': expanded_mask.astype(bool),
                'area': expanded_mask.sum(),
                'bbox': list(cv2.boundingRect(expanded_mask.astype(np.uint8))),
            })
            if masked_area is None:
                masked_area = expanded_mask.astype(np.uint8)
            else:
                masked_area[expanded_mask] += 1

        non_masked_area = masked_area == 0
        labeled_mask, num_labels = label(non_masked_area)
        
        for i in range(1, num_labels + 1):
            post_processed_masks.append({
                'mask': labeled_mask == i,
                'area': (labeled_mask == i).sum(),
                'bbox': list(cv2.boundingRect((labeled_mask == i).astype(np.uint8))),
            })
        return post_processed_masks