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

import pprint
pp = pprint.PrettyPrinter(width=128, compact=True)
import os

DEVICE = "cuda"

import sys
sys.path.append('segmentation/')
from segmentation import Segmenter, visualized_masks


import datasets
from data.sharegpt4v import ShareGPT4V
from data.clevr import CLEVR

from transformers import CodeGenTokenizerFast
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import pycocotools.mask as mask_util
import pickle
from model.multimodal_tokenizer import MultimodalTokenizer

from transformers import (
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, PeftModel, get_peft_model

# Patch 32x32
patch_size = 32
max_seg_per_img = 150  
square_patches = True
ckpt = "/home/dchenbs/workspace/Seq2Seq-AutoEncoder/runs/phi-2-multimodal/clevr-patchlmm(32)-10ep-lora(64)-bs32-lr1e-4-LinearAE/checkpoint-21000"

# # Patch 80x80
# patch_size = 80
# max_seg_per_img = 24 
# square_patches = True
# ckpt = "/home/dchenbs/workspace/Seq2Seq-AutoEncoder/runs/phi-2-multimodal/patchlmm-80-clevr-full-5ep-lora(32)-bs64-lr1e-4/checkpoint-2000"

# # segment seqae-based
# square_patches = False
# max_seg_per_img = 32
# ckpt = "/home/dchenbs/workspace/Seq2Seq-AutoEncoder/runs/phi-2-multimodal/clevr-seglmm-10ep-lora(64)-bs32-lr1e-4/checkpoint-7000"


if square_patches:
    segmenter = Segmenter(model_name='square_patches', patch_size=patch_size, do_mask_expansion=False)
else:
    segmenter = None

# with open('segmentation/mobile_sam_v2_l2.json', 'r') as f:
#     config = json.load(f)
#     pprint.pprint(config)
# segmenter = Segmenter(config['model_name'], config['checkpoint'], **config['kwargs'])


from model.seq2seq_autoencoder import Seq2SeqAutoEncoderModel
from model.modeling_multimodal_phi import PhiForMultimodalModeling, load_seqae

model = PhiForMultimodalModeling.from_pretrained(
    "microsoft/phi-2",
    w_segment_loss=1.0,
    w_bbox_loss=1/(1000*1000),
    seqae_batch_size=32,
    )

seqae = Seq2SeqAutoEncoderModel.from_pretrained('/home/dchenbs/workspace/Seq2Seq-AutoEncoder/runs/Jan02_11-49-33_host19-SA1B-[327MB-16queries-1024]-[lr1e-05-bs16x1step-8gpu]/checkpoints/checkpoint_step4350k')
seqae.enable_caching_latents(
    cache_dir='segmentation/cached_segments/seqae_cache_dir',
    use_existing=True
    )
load_seqae(model, seqae)

peft_model = PeftModel.from_pretrained(model, ckpt, from_transformers=True)

state_dict = torch.load(ckpt+'/additional_modules.pth', map_location="cpu")
message = peft_model.load_state_dict(state_dict, strict=False)

model = peft_model.merge_and_unload()
model = model.to(DEVICE)

tokenizer = MultimodalTokenizer.from_pretrained(
    "microsoft/phi-2", trust_remote_code=True, 
    segmenter=segmenter, 
    seqae_config=model.seqae.config, 
    max_seg_per_img=max_seg_per_img,
    model_max_length=max_seg_per_img+100,
    )

if not square_patches:
    tokenizer.load_cached_segments(
        # '/home/dchenbs/workspace/Seq2Seq-AutoEncoder/segmentation/cached_segments/sharegpt4v_instruct_gpt4-vision_cap100k')
        '/home/dchenbs/workspace/Seq2Seq-AutoEncoder/segmentation/cached_segments/clevr_val')

model.special_token_id_mapping = {
    "<|startofimage|>": tokenizer.convert_tokens_to_ids("<|startofimage|>"),
    "<|endofimage|>": tokenizer.convert_tokens_to_ids("<|endofimage|>"),
    "<|seg|>": tokenizer.convert_tokens_to_ids("<|seg|>"),
    "<|endoftext|>": tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    "[PAD]": tokenizer.convert_tokens_to_ids("[PAD]"),
}

root = '/home/dchenbs/workspace/datasets/CLEVR_v1.0/images/val'
images = os.listdir(root)
image_path = os.path.join(root, random.choice(images))

plt.figure(figsize=(30, 10))
image = Image.open(image_path)

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('original')
plt.axis('off')

inputs  = tokenizer([f'<img_path>{image_path}</img_path>'], return_tensors="pt")

canvas = tokenizer.decode_image(inputs['segment_sequences'][0][0], inputs['bboxes'][0][0])
plt.subplot(1, 3, 2)
plt.imshow(canvas)
plt.title('reconstructed')
plt.axis('off')

canvas = tokenizer.decode_image(inputs['segment_sequences'][0][0], inputs['bboxes'][0][0], filling='random')
plt.subplot(1, 3, 3)
plt.imshow(canvas)
plt.axis('off')
plt.title('segment masks')
plt.show()


# dataset_path = '/home/dchenbs/workspace/datasets/sharegpt4v/ShareGPT4V'
# dataset_identifier = 'sharegpt4v_instruct_gpt4-vision_cap100k'

dataset_path = '/home/dchenbs/workspace/datasets/CLEVR_v1.0'
dataset_identifier = 'clevr_val'

if 'sharegpt4v' in dataset_identifier:
    annotation_file = os.path.join(dataset_path, dataset_identifier+'.json')
    dataset = ShareGPT4V(annotation_file, split='val')
    failed_samples = dataset.validate_exist(valid_img_paths=tokenizer.cache)
    
elif 'clevr' in dataset_identifier:
    split = dataset_identifier.split('_')[-1]
    dataset = CLEVR(dataset_path=dataset_path, split=split)
    # failed_samples = dataset.validate_exist(valid_img_paths=tokenizer.cache)

    all_gt = []
all_pred = []
all_prompt = []


for i in tqdm.tqdm(range(2000)):
    sample_idx = i

    prompt, target = dataset[sample_idx].split('### AI: \n')

    inputs  = tokenizer([
        prompt
        ], return_tensors="pt", return_attention_mask=False, padding=True)

    inputs['input_ids'] = inputs['input_ids'].to(DEVICE)
    inputs['segment_sequences'] = [segment_sequence.to(DEVICE) for segment_sequence in inputs['segment_sequences']]
    inputs['bboxes'] = [bbox.to(DEVICE) for bbox in inputs['bboxes']]

    outputs = model.generate(
        # **inputs, 
        input_ids=inputs['input_ids'],
        segment_sequences=inputs['segment_sequences'],
        bboxes=inputs['bboxes'],
        use_cache=False,
        max_length=512)
    text = tokenizer.batch_decode(outputs)[0]
    
    # img_path = dataset.__getitem__(sample_idx, only_return_img_path=True)
    # img = Image.open(img_path)
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()
    # print(text)
    # print(f'### GT:\n{target}')

    all_gt.append(target)
    all_pred.append(text.split('### AI: \n')[1])
    all_prompt.append(prompt)

import re

def parse_caption(caption):

    # example caption: Total 5 objects: a large brown rubber cylinder, a small green rubber cylinder, a large gray rubber cube, a large purple metal sphere, a small gray metal cube.<|endoftext|>

    count, objects_size, objects_color, objects_material, objects_shape = None, None, None, None, None

    try:
        count = int(re.search(r'Total (\d+) objects', caption).group(1))
        objects = re.findall(r'a (\w+) (\w+) (\w+) (\w+)', caption)

        objects_size = [obj[0] for obj in objects]
        objects_color = [obj[1] for obj in objects]
        objects_material = [obj[2] for obj in objects]
        objects_shape = [obj[3] for obj in objects]
        success = True
    except:
        success = False

    return success, count, objects_size, objects_color, objects_material, objects_shape

def evaluate_clevr(all_gt, all_pred):
    # get number of parsable predicted captions (success rate)
    # get accuracy of each attribute

    parsable_captions = 0
    parsable_objects = 0

    counting_correct = 0
    size_correct = 0
    color_correct = 0
    material_correct = 0
    shape_correct = 0
    
    for gt, pred in zip(all_gt, all_pred):
        _, num_objs, objects_size, objects_color, objects_material, objects_shape = parse_caption(gt)
        success, pred_num_objs, pred_objects_size, pred_objects_color, pred_objects_material, pred_objects_shape = parse_caption(pred)

        # print(_, num_objs, objects_size, objects_color, objects_material, objects_shape)
        # print(success, pred_num_objs, pred_objects_size, pred_objects_color, pred_objects_material, pred_objects_shape)
        
        if success:
            parsable_captions += 1
            if num_objs == pred_num_objs:
                counting_correct += 1
            for i in range((min(len(pred_objects_size), len(objects_size)))):
                if objects_size[i] == pred_objects_size[i]:
                    size_correct += 1
                if objects_color[i] == pred_objects_color[i]:
                    color_correct += 1
                if objects_material[i] == pred_objects_material[i]:
                    material_correct += 1
                if objects_shape[i] == pred_objects_shape[i]:
                    shape_correct += 1
                parsable_objects += 1

    parsable_rate = parsable_captions / len(all_gt)
    counting_acc = counting_correct / parsable_captions
    size_acc = size_correct / parsable_objects
    color_acc = color_correct / parsable_objects
    material_acc = material_correct / parsable_objects
    shape_acc = shape_correct / parsable_objects

    return {
        'evaluated_captions': len(all_gt),
        'parsable_rate': parsable_rate,
        'counting_acc': counting_acc,
        'size_acc': size_acc,
        'color_acc': color_acc,
        'material_acc': material_acc,
        'shape_acc': shape_acc
    }

evaluation = evaluate_clevr(all_gt, all_pred)
print(evaluation)
print(ckpt)

predictions = []
for gt, pred, promt in zip(all_gt, all_pred, all_prompt):
    predictions.append({
        'prompt': promt,
        'ground_truth': gt,
        'prediction': pred
    })

results = {
    'evaluation': evaluation,
    'predictions': predictions,
}

json.dump(results, open(ckpt+'/evaluations.json', 'w'), indent=4)