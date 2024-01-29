import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(0)
import os
from scipy.ndimage import label
import time
import pprint
import torch
import tqdm

from PIL import Image
from typing import Any, Dict, Generator,List
from segment_anything.utils.amg import batched_mask_to_box

class Segmenter():

    def __init__(
            self, 
            model_name,
            checkpoint,

            # SAM, FastSAM, MobileSAM, RepViT-SAM
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

            # MobileSAMv2
            num_box_prompts = 320,
            object_conf = 0.4,
            object_iou = 0.9,

            device = 'cuda',
            ):
        self.generator = None
        self.model_name = model_name

        if self.model_name=='sam':
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            model_type = checkpoint.split('/')[-1][4:9]
            sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device).eval()
        
        elif self.model_name=='fast_sam':
            # Fast Segment Anything 
            # https://arxiv.org/abs/2306.12156 (21 Jun 2023)
            # https://github.com/CASIA-IVA-Lab/FastSAM

            from fastsam import FastSAM
            self.generator = FastSAM(checkpoint)

        elif self.model_name=='mobile_sam':
            # Faster Segment Anything: Towards Lightweight SAM for Mobile Applications 
            # https://arxiv.org/abs/2306.14289.pdf (25 Jun 2023)
            # https://github.com/ChaoningZhang/MobileSAM
            
            from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
            sam = sam_model_registry["vit_t"](checkpoint=checkpoint).to(device).eval()

        elif self.model_name=='repvit_sam':
            # RepViT-SAM: Towards Real-Time Segmenting Anything
            # https://arxiv.org/abs/2312.05760 (10 Dec 2023)
            # https://github.com/THU-MIG/RepViT

            from repvit_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
            sam = sam_model_registry["repvit"](checkpoint=checkpoint).to(device).eval()

        elif self.model_name=='mobile_sam_v2':
            # MobileSAMv2: Faster Segment Anything to Everything
            # https://arxiv.org/abs/2312.09579 (15 Dec 2023)
            # https://github.com/ChaoningZhang/MobileSAM/tree/master/MobileSAMv2/mobilesamv2

            from mobilesamv2.promt_mobilesamv2 import ObjectAwareModel
            from mobilesamv2 import sam_model_registry, SamPredictor

            ckpt_to_model_type = {
                'l2.pt': 'efficientvit_l2',
                'mobile_sam.pt': 'tiny_vit',
                'sam_vit_h.pt': 'sam_vit_h',
            }

            self.num_box_prompts = num_box_prompts
            self.object_conf = object_conf
            self.object_iou = object_iou

            obj_model_path = os.path.join(os.path.dirname(checkpoint), 'ObjectAwareModel.pt')
            prompt_guided_path = os.path.join(os.path.dirname(checkpoint), 'Prompt_guided_Mask_Decoder.pt')

            ObjAwareModel = ObjectAwareModel(obj_model_path)
            PromptGuidedDecoder=sam_model_registry['PromptGuidedDecoder'](prompt_guided_path)
            
            sam = sam_model_registry['vit_h']()
            sam.prompt_encoder=PromptGuidedDecoder['PromtEncoder']
            sam.mask_decoder=PromptGuidedDecoder['MaskDecoder']
            sam.image_encoder=sam_model_registry[ckpt_to_model_type[checkpoint.split('/')[-1]]](checkpoint)
            
            sam = sam.to(device).eval()
            self.generator = SamPredictor(sam)
            self.generator.ObjAwareModel = ObjAwareModel

            # for bluring in post prosessing
            kernel_size = 7
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
            self.kernel = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0).to('cuda')

        else:
            raise NotImplementedError(f'Model {self.model_name} not implemented')
        
        if self.generator is None:
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
        
    @torch.no_grad()
    def __call__(self, image_path, resize_to_max_of=-1, profiling=False):  
        profile = {}
        image = np.array(Image.open(image_path).convert('RGB'))
        
        if resize_to_max_of == -1 or (image.shape[0] <= resize_to_max_of and image.shape[1] <= resize_to_max_of):
            height, width = image.shape[:2]
        else:
            if image.shape[0] > image.shape[1]:
                height, width = resize_to_max_of, int(resize_to_max_of * image.shape[1] / image.shape[0])
            else:
                height, width = int(resize_to_max_of * image.shape[0] / image.shape[1]), resize_to_max_of
            image = cv2.resize(image, (width, height))

        
        if self.model_name=='fast_sam':
            image = Image.fromarray(image)
            everything_results = self.generator(image, device='cuda', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
            print(everything_results)
            masks = []
            for i in range(everything_results[0].boxes.data.shape[0]):
                box = everything_results[0].boxes.data[i]
                mask = everything_results[0].masks.data[i]
                masks.append({'segmentation': mask.astype(bool), 'area': mask.sum(), 'bbox': box.cpu().tolist(),})
        
        elif self.model_name=='mobile_sam_v2':
            
            start = time.time()
            obj_results = self.generator.ObjAwareModel(
                image, device='cuda',
                retina_masks=True,
                imgsz=1024,
                conf=self.object_conf,
                iou=self.object_iou,
                )
            profile['object detection'] = time.time()-start
            start = time.time()

            self.generator.set_image(image)

            profile['sam.set_image(image)'] = time.time()-start
            start = time.time()
            
            input_boxes1 = obj_results[0].boxes.xyxy
            input_boxes = input_boxes1.cpu().numpy()
            input_boxes = self.generator.transform.apply_boxes(input_boxes, self.generator.original_size)
            input_boxes = torch.from_numpy(input_boxes).cuda()

            sam_mask=[]
            image_embedding=self.generator.features
            image_embedding=torch.repeat_interleave(image_embedding, self.num_box_prompts, dim=0)
            prompt_embedding=self.generator.model.prompt_encoder.get_dense_pe()
            prompt_embedding=torch.repeat_interleave(prompt_embedding, self.num_box_prompts, dim=0)

            for (boxes,) in batch_iterator(self.num_box_prompts, input_boxes):
                with torch.no_grad():
                    image_embedding=image_embedding[0:boxes.shape[0],:,:,:]
                    prompt_embedding=prompt_embedding[0:boxes.shape[0],:,:,:]
                    sparse_embeddings, dense_embeddings = self.generator.model.prompt_encoder(
                        points=None,
                        boxes=boxes,
                        masks=None,)
                    low_res_masks, _ = self.generator.model.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=prompt_embedding,
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                        simple_type=True,
                    )
                    low_res_masks=self.generator.model.postprocess_masks(low_res_masks, self.generator.input_size, self.generator.original_size)
                    sam_mask_pre = (low_res_masks > self.generator.model.mask_threshold)*1.0
                    sam_mask.append(sam_mask_pre.squeeze(1))
            
            masks = torch.cat(sam_mask)
            profile['mask generation'] = time.time()-start
            start = time.time()

        else:
            masks = self.generator.generate(image)
            for mask in masks:
                if type(mask['segmentation']) != torch.Tensor:
                    mask['segmentation'] = torch.from_numpy(mask['segmentation'])
            masks = torch.stack([mask['segmentation'] for mask in masks]).to('cuda')

        masks = self.post_processing_masks(masks)
        profile['postproceessing'] = time.time()-start
        
        if profiling:
            return masks, profile
        else:
            return masks
        
    def post_processing_masks(self, masks):
        h, w = masks.shape[1:]
        masks = masks.unsqueeze(1)
        masks = torch.nn.functional.conv2d(masks, self.kernel, padding=self.kernel.shape[2]//2)[:,0].bool().cpu()

        boxes = batched_mask_to_box(masks)
        # convert xyxy to xywh
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        masks = masks.numpy()
        masked_area = None
        post_processed_masks = []
        for i in range(len(masks)):
            post_processed_masks.append({
                'segmentation': masks[i],
                'area': int(masks[i].sum()),
                'bbox': boxes[i].numpy().tolist(),
                # 'bbox': list(cv2.boundingRect(expanded_mask_uint8)),
            })
            if masked_area is None:
                masked_area = masks[0].astype(np.uint8)
            else:
                masked_area[masks[0]] += 1

        non_masked_area = masked_area == 0
        labeled_mask, num_labels = label(non_masked_area)

        labeled_mask = torch.from_numpy(labeled_mask)
        masks = torch.zeros((num_labels, h, w), dtype=torch.bool)
        for i in range(1, num_labels + 1):
            mask = torch.where(labeled_mask == i, True, False)
            masks[i-1] = mask

        boxes = batched_mask_to_box(masks)
        # convert xyxy to xywh
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        masks = masks.numpy()
        for i in range(num_labels):
            post_processed_masks.append({
                'segmentation': masks[i],
                'area': int(masks[i].sum()),
                'bbox': boxes[i].numpy().tolist(),
                # 'bbox': list(cv2.boundingRect(masks[i].astype(np.uint8))),
            })

        return post_processed_masks


def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]

    
def visualized_masks(masks, image):
    canvas = np.ones_like(image) * 255
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    for mask in masks:
        if type(mask['segmentation']) == torch.Tensor:
            mask['segmentation'] = mask['segmentation'].cpu().numpy()
        average_color = np.mean(image[mask['segmentation'] == 1], axis=0)
        canvas[mask['segmentation'] == 1] = average_color

        # visualize segment boundary
        contours, _ = cv2.findContours(mask['segmentation'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (200, 200, 200), 1)

    return canvas


def get_masked_area(masks):
    masked_area = None
    for mask in masks:
        if masked_area is None:
            masked_area = mask['segmentation'].astype(np.uint8)
        else:
            masked_area[mask['segmentation']] += 1

    non_masked_area = (masked_area == 0).astype(np.uint8)
    return masked_area, non_masked_area
