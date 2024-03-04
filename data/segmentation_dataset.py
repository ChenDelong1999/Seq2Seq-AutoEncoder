import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.mask import decode

from scipy.ndimage import label
from torchvision import transforms
import os
import json
import random

try:
    from lvis import LVIS
    import cv2
except:
    pass # temporary fix for the lvis & opencv installation issue

np.random.seed(1)
random.seed(1)

def get_bounding_box(mask):
    y_indices, x_indices = np.where(mask)
    x_min = np.min(x_indices)
    y_min = np.min(y_indices)
    x_max = np.max(x_indices)
    y_max = np.max(y_indices)
    w = x_max - x_min
    h = y_max - y_min
    w = 1 if w == 0 else w
    h = 1 if h == 0 else h
    return x_min, y_min, w, h


class ClevrPatchDataset:
    def __init__(self, clevr_root, patch_size):
        self.dataset_name = 'clevr_patch'
        self.clevr_root = clevr_root
        self.patch_size = patch_size
        self.img_paths = []
        
        for img_file in os.listdir(clevr_root):
            self.img_paths.append(os.path.join(clevr_root, img_file))
        print(f"clevr_patch dataset loaded. {len(self.img_paths)} images found.")

        self.num_segments = 1000000000
        self.num_images = len(self.img_paths)
        self.num_categories = 1

        random.shuffle(self.img_paths)

    def get_all_captions(self):
        return []
    
    def load_segments_from_one_image(self):
        success = False
        while not success:
            img_path = self.img_paths[np.random.randint(0, len(self.img_paths))]
            try:
                image = np.array(Image.open(img_path).convert('RGB'))
                success = True
            except:
                continue

        segments = []
        h, w = image.shape[:2]
        n_h = h // self.patch_size
        n_h += 1 if h % self.patch_size != 0 else 0

        n_w = w // self.patch_size
        n_w += 1 if w % self.patch_size != 0 else 0

        n_patches = n_h * n_w
        mask = np.ones((self.patch_size, self.patch_size), dtype=bool)

        for i in range(n_h):
            for j in range(n_w):
                y = i * self.patch_size
                x = j * self.patch_size
                # masks[i*n_w+j, y:y+self.patch_size+1, x:x+self.patch_size+1] = True
                # mask[y:y+self.patch_size+1, x:x+self.patch_size+1] = True
                segments.append({
                    "patch": image[y:y+self.patch_size+1, x:x+self.patch_size+1],
                    "mask": mask,
                    'image_path': img_path,
                    'bbox': [x, y, self.patch_size, self.patch_size], # 'x', 'y', 'width', 'height
                    "name": "",
                    "caption": "",
                })

        return segments
    
    def class_name_to_class_id(self, name):
        return 0


class SA1BDataset:
    def __init__(self, sa1b_root):
        self.dataset_name = 'sa1b'
        self.sa1b_root = sa1b_root
        self.img_ids = []
        
        for subfolder in os.listdir(sa1b_root):
            subfolder_path = os.path.join(sa1b_root, subfolder)
            if os.path.isdir(subfolder_path):
                for img_file in os.listdir(subfolder_path):
                    if img_file.endswith('.jpg'):
                        self.img_ids.append(os.path.join(subfolder_path, img_file[:-4]))
        print(f"SA1B dataset loaded. {len(self.img_ids)} images found.")

        self.num_segments = 1000000000
        self.num_images = len(self.img_ids)
        self.num_categories = 1

        random.shuffle(self.img_ids)

    def get_all_captions(self):
        return []

    def preprocess_annotations(self, annotations, split_disconnected, min_pixel_num=16):
        new_annotations = []
        for annotation in annotations:
            annotation['segmentation'] = decode(annotation['segmentation'])
            if not split_disconnected:
                new_annotations.append(annotation)
            else:
                labeled_mask, num_labels = label(annotation['segmentation'])
                for i in range(1, num_labels + 1):
                    new_annotation = annotation.copy()
                    new_annotation['segmentation'] = (labeled_mask == i)
                    if new_annotation['segmentation'].sum() >= min_pixel_num:
                        new_annotation['bbox'] = get_bounding_box(new_annotation['segmentation'])
                        new_annotations.append(new_annotation)
        return new_annotations
    
    def load_segments_from_one_image(self):
        success = False
        while not success:
            img_id = self.img_ids[np.random.randint(0, len(self.img_ids))]
            try:
                image = np.array(Image.open(img_id + '.jpg').convert('RGB'))
                annotations = json.load(open(img_id + '.json'))['annotations']
                annotations = self.preprocess_annotations(annotations, split_disconnected=False)
                success = True
            except:
                continue

        segments = []
        for annotation in annotations:
            bbox = [int(b) for b in annotation['bbox']]
            bbox[2] = 1 if bbox[2] == 0 else bbox[2]
            bbox[3] = 1 if bbox[3] == 0 else bbox[3]
            x, y, w, h = bbox
            patch = image[y:y+h, x:x+w]

            mask = annotation['segmentation']
            patch_mask = mask[y:y+h, x:x+w]
            segments.append({
                "patch": patch,
                "mask": patch_mask,
                'image_path': img_id + '.jpg',
                'bbox': bbox, # 'x', 'y', 'width', 'height
                "name": "",
                "caption": "",
            })  

        return segments
    
    def class_name_to_class_id(self, name):
        return 0


class VisualGenomeDataset:
    def __init__(self, visual_genome_root, split):
        self.dataset_name = 'visual_genome'
        self.visual_genome_root = visual_genome_root
        self.split = split
        self.annotations_path = os.path.join(visual_genome_root, f'visual_genome_1117_sam_mask.jsonl')
        self.file = open(self.annotations_path, 'r')

        self.num_segments = 0
        images = []
        for line in self.file:
            sample = json.loads(line)
            self.num_segments += len(sample['objects'])
            images.append(sample['image'])
        self.num_images = len(set(images))
        self.file.seek(0)
        self.num_categories = 1

    def get_all_captions(self):
        captions = []
        for line in self.file:
            objects = json.loads(line)['objects']
            for obj in objects:
                caption = obj['caption']
                captions.append(caption)

        captions = list(set(captions))
        return captions

    def load_segments_from_one_image(self):
        line = self.file.readline()
        if not line:
            self.file.seek(0)
            line = self.file.readline()
        sample = json.loads(line)

        image_path = os.path.join(self.visual_genome_root, sample['image'])
        image = np.array(Image.open(image_path).convert('RGB'))
        segments = []
        for obj in sample['objects']:
            mask = decode(obj['mask'])
            if mask.sum() == 0:
                continue
            bbox = get_bounding_box(mask)
            x, y, w, h = bbox
            patch = image[y:y+h, x:x+w]
            patch_mask = mask[y:y+h, x:x+w]

            segments.append({
                "patch": patch,
                "mask": patch_mask,
                'image_path': image_path,
                'bbox': bbox,
                "label": 0,
                "name": obj['name'],
                "caption": obj['caption'],
            })

        return segments
    
    def class_name_to_class_id(self, name):
        return 0
       

class V3DetDataset:
    def __init__(self, v3det_root, split):
        self.dataset_name = 'v3det'
        self.v3det_root = v3det_root
        self.split = split
        self.annotations_path = os.path.join(v3det_root, f'annotations/v3det_20231116_sam_masks_{split}.jsonl')
        self.images_path = os.path.join(v3det_root, f'images')
        
        self.category_info = json.load(open(os.path.join(v3det_root, 'annotations/v3det_2023_v1_category_info.json')))
        self.num_categories = len(self.category_info)

        self.num_images = 0
        self.num_segments = 0
        self.samples = []
        for line in open(self.annotations_path, 'r'):
            sample = json.loads(line)
            self.samples.append(sample)
            self.num_segments += len(sample['objects'])
            self.num_images += 1
        self.sample_index = -1


    def get_all_captions(self):
        captions = []
        for cat in self.category_info.values():
            caption = f'An image of a {cat["name"]}, {cat["cat_info"]}'
            captions.append(caption)
        return captions

    def load_segments_from_one_image(self):
        if self.sample_index >= len(self.samples):
            self.sample_index = -1
        self.sample_index += 1
        sample = self.samples[self.sample_index]

        image_path = os.path.join(self.images_path, sample['image'])
        image = np.array(Image.open(image_path).convert('RGB'))
        objects = sample['objects']
        segments = []
        for obj in objects:
            mask = decode(obj['mask'])
            if mask.sum() == 0:
                continue
            bbox = get_bounding_box(mask)
            
            x, y, w, h = bbox
            patch = image[y:y+h, x:x+w]
            patch_mask = mask[y:y+h, x:x+w]

            category_info = self.category_info[obj['name']]
            caption = f'An image of a {category_info["name"]}, {category_info["cat_info"]}'# {category_info["cat_info_gpt"]}'

            segments.append({
                "patch": patch,
                "mask": patch_mask,
                'image_path': image_path,
                'bbox': bbox,
                "label": 0,
                "name": obj['name'],
                "caption": caption,
            })

        return segments
    
    def class_name_to_class_id(self, name):
        return self.category_info[name]['id']
        

class LVISDataset:
    def __init__(self, lvis_root, coco_root, split):
        self.dataset_name = 'lvis'
        self.lvis_root = lvis_root
        self.coco_root = coco_root

        if split == 'train':
            self.sub_dir = 'train2017'
            self.num_segments = 1270141
        elif split == 'val':
            self.sub_dir = 'val2017'
            self.num_segments = 244707
        else:
            raise NotImplementedError

        self.lvis = LVIS(f'{self.lvis_root}/lvis_v1_{split}.json')
        self.img_ids = self.lvis.get_img_ids()
        self.load_anns = self.lvis.load_anns()
        self.num_images = len(self.img_ids)

        self.class_name_to_class_id_mapping = {}
        for cat in self.lvis.cats.values():
            name = ', '.join([synonym.replace('_', ' ') for synonym in cat['synonyms']]) 
            self.class_name_to_class_id_mapping[name] = cat['id']
        self.num_categories = len(self.class_name_to_class_id_mapping)
        self.image_index = -1

    def get_all_captions(self):
        captions = []
        for cat in self.lvis.cats.values():
            name = ', '.join([synonym.replace('_', ' ') for synonym in cat['synonyms']]) 
            caption = f'An image of a {name}, {cat["def"]}'
            captions.append(caption)
        return captions
        
    def preprocess_annotations(self, annotations, split_disconnected, min_pixel_num=16):
        new_annotations = []
        for annotation in annotations:
            if type(annotation['segmentation']) == list:
                annotation['segmentation'] = self.lvis.ann_to_mask(annotation)
            if not split_disconnected:
                new_annotations.append(annotation)
            else:
                labeled_mask, num_labels = label(annotation)
                for i in range(1, num_labels + 1):
                    new_annotation = annotation.copy()
                    new_annotation['segmentation'] = (labeled_mask == i)
                    if new_annotation['segmentation'].sum() >= min_pixel_num:
                        new_annotation['bbox'] = get_bounding_box(new_annotation['segmentation'])
                        new_annotations.append(new_annotation)
        return new_annotations

    def load_segments_from_one_image(self):
        # image_index = np.random.randint(0, len(self.img_ids))
        if self.image_index >= len(self.img_ids):
            self.image_index = -1
        self.image_index += 1
        img_id = self.img_ids[self.image_index]

        img = self.lvis.load_imgs([img_id])[0]
        img_path = img['coco_url'].replace('http://images.cocodataset.org', os.path.join(self.coco_root, 'images'))
        image = np.array(Image.open(img_path).convert('RGB'))

        annotations = self.lvis.load_anns(self.lvis.get_ann_ids([img_id]))
        annotations = self.preprocess_annotations(annotations, split_disconnected=False)
        segments = []
        for annotation in annotations:
            label = annotation['category_id']
            mask = annotation['segmentation']
            bbox = [int(b) for b in annotation['bbox']]
            bbox[2] = 1 if bbox[2] == 0 else bbox[2]
            bbox[3] = 1 if bbox[3] == 0 else bbox[3]
            
            # masked_img = image * mask[:, :, None]
            x, y, w, h = bbox
            patch = image[y:y+h, x:x+w]
            patch_mask = mask[y:y+h, x:x+w]

            category = self.lvis.load_cats([label])[0]
            name = ', '.join([synonym.replace('_', ' ') for synonym in category['synonyms']]) 
            caption = f'An image of a {name}, {category["def"]}'

            segments.append({
                "patch": patch,
                "mask": patch_mask,
                'image_path': img_path,
                'bbox': bbox, # 'x', 'y', 'width', 'height
                "name": name,
                "caption": caption,
            })  

        return segments

    def class_name_to_class_id(self, name):
        return self.class_name_to_class_id_mapping[name]


class COCODataset:
    def __init__(self, coco_root, split, text_features=None):
        self.dataset_name = 'coco'
        self.coco_root = coco_root
        if split == 'train':
            self.sub_dir = 'train2017'
            self.num_segments = 860001
        elif split == 'val':
            self.sub_dir = 'val2017'
            self.num_segments = 36781
        else:
            raise NotImplementedError

        self.coco = COCO(f'{self.coco_root}/annotations/instances_{self.sub_dir}.json')
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        self.num_images = len(self.img_ids)
        self.num_categories = len(self.coco.cats)
        self.id_to_name = {category['id']: category['name'] for category in self.coco.loadCats(self.coco.getCatIds())}
        self.image_index = -1

    def get_all_captions(self):
        captions = [f'An image of a {self.id_to_name[category_id]}' for category_id in self.id_to_name.keys()]
        return captions

    def preprocess_annotations(self, annotations, min_pixel_num):
        new_annotations = []
        for annotation in annotations:
            mask = self.coco.annToMask(annotation)
            labeled_mask, num_labels = label(mask)
            for i in range(1, num_labels + 1):
                new_annotation = annotation.copy()
                new_annotation['segmentation'] = (labeled_mask == i)
                if new_annotation['segmentation'].sum() >= min_pixel_num:
                    new_annotation['bbox'] = [
                        np.min(np.where(new_annotation['segmentation'])[1]),  # x_min
                        np.min(np.where(new_annotation['segmentation'])[0]),  # y_min
                        np.max(np.where(new_annotation['segmentation'])[1]) - np.min(np.where(new_annotation['segmentation'])[1]),  # width
                        np.max(np.where(new_annotation['segmentation'])[0]) - np.min(np.where(new_annotation['segmentation'])[0]),  # height
                    ]
                    new_annotations.append(new_annotation)
        return new_annotations

    def load_segments_from_one_image(self):
        # image_index = np.random.randint(0, len(self.img_ids))
        if self.image_index >= len(self.img_ids):
            self.image_index = -1
        self.image_index += 1
        image_index = self.image_index

        img_path = f"{self.coco_root}/images/{self.sub_dir}/{self.coco.loadImgs(self.img_ids[image_index])[0]['file_name']}"
        image = np.array(Image.open(img_path).convert('RGB'))

        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=self.img_ids[image_index]))
        annotations = self.preprocess_annotations(annotations, min_pixel_num=16)
        segments = []
        for annotation in annotations:
            label = annotation['category_id']
            mask = annotation['segmentation']
            bbox = [int(b) for b in annotation['bbox']]
            bbox[2] = 1 if bbox[2] == 0 else bbox[2]
            bbox[3] = 1 if bbox[3] == 0 else bbox[3]
            
            # masked_img = image * mask[:, :, None]
            x, y, w, h = bbox
            patch = image[y:y+h, x:x+w]
            patch_mask = mask[y:y+h, x:x+w]
            segments.append({
                "patch": patch,
                "mask": patch_mask,
                'image_path': img_path,
                'bbox': bbox, # 'x', 'y', 'width', 'height
                "label": label,
                "name": self.id_to_name[label],
                "caption": f'An image of a {self.id_to_name[label]}',
            })  

        return segments
    
    def class_name_to_class_id(self, name):
        return self.coco.getCatIds(catNms=[name])[0]


class SeqMaskDataset(Dataset):
    def __init__(self, dataset, num_queries, data_seq_length=64*64, min_resize_ratio=1.0, text_features=None, expand_mask_ratio=0):
        self.dataset = dataset
        self.virtual_dataset_size = self.dataset.num_segments
        print(f'Dataset {self.dataset.dataset_name} has {self.dataset.num_images} images and {self.virtual_dataset_size} segment instances in {self.dataset.num_categories} categories.')
        self.num_queries = num_queries
        self.data_seq_length = data_seq_length
        self.model_seq_length = self.data_seq_length + self.num_queries + 1
        self.num_channels = 3 + 1 + 1
        self.img_channels = 3
        self.channel_info = {
                'data': [0,3],
                'shape_encoding': 3,
                'is_data': 4,
            }
        self.data_seq_length_multiplier = 1
        self.sample_buffer = []

        self.min_resize_ratio = min_resize_ratio
        self.expand_mask_ratio = expand_mask_ratio

        if text_features is not None:
            self.text_features = np.load(text_features, allow_pickle=True).item()
            print(f'Loaded {len(self.text_features.keys())} text features from "{text_features}".')
            for caption in self.dataset.get_all_captions():
                if caption not in self.text_features.keys():
                    raise ValueError(f'Caption "{caption}" not found in "{text_features}".')
        else:
            print('No text features provided. Will not use text features.')
            self.text_features = None

    def __getitem__(self, index):
        while len(self.sample_buffer) == 0:
            segments = self.dataset.load_segments_from_one_image()
            for segment in segments:
                segment = self.resize(segment)
                if self.expand_mask_ratio > 0:
                    try:
                        segment = self.expand_mask(segment, self.expand_mask_ratio)
                    except:
                        print(segment)
                segment['patch'] = segment['patch'] * segment['mask'][:, :, None]
                segment = self.encode_to_sequence(segment)
                self.sample_buffer.append(segment)
           
        segment = self.sample_buffer.pop()
        segment_info = {
            'width': segment['patch'].shape[1],
            'height': segment['patch'].shape[0],
            'name': segment['name'],
            'caption': segment['caption'],
            'image_path': segment['image_path'],
            'bbox': segment['bbox'],
            'text_feature': self.text_features[segment['caption']] if self.text_features is not None else [],
        }
        return segment['data_sequence'], segment_info

    def __len__(self):
        return self.virtual_dataset_size

    def update_data_seq_length_multiplier(self, data_seq_length_multiplier):
        self.data_seq_length_multiplier = data_seq_length_multiplier
    
    def expand_mask(self, segment, expand_mask_ratio):
        kernel_size = int(min(segment['patch'].shape[:2]) * expand_mask_ratio) // 2 * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
        segment['mask'] = segment['mask'].astype(np.uint8)
        blurred_mask = cv2.filter2D(segment['mask'],-1,kernel)
        segment['mask'] = (blurred_mask > 0).astype(bool)
        return segment

    def resize(self, segment):
        h, w = segment['patch'].shape[:2]
        if h * w > (self.data_seq_length * self.data_seq_length_multiplier):
            ratio_to_maxlength = np.sqrt((self.data_seq_length * self.data_seq_length_multiplier) / (h * w))
            ratio_random_resize = np.random.uniform(self.min_resize_ratio, 1.0)
            h = int(h * ratio_to_maxlength * ratio_random_resize)
            w = int(w * ratio_to_maxlength * ratio_random_resize)
            segment['patch'] = np.array(Image.fromarray(segment['patch']).resize((w, h)))
            segment['mask'] = np.array(Image.fromarray(segment['mask']).resize((w, h)))
        return segment

    def encode_to_sequence(self, segment):
        h, w = segment['patch'].shape[:2]
        sequence = []
        for i in range(h):
            for j in range(w):
                pixel_data = segment['patch'][i, j] / 255.0
                is_rightmost = 1 if j == w - 1 else 0
                is_non_masked = int(segment['mask'][i, j])
                sequence.append(pixel_data.tolist() + [is_rightmost, is_non_masked])
        sequence = np.array(sequence) 

        # pad the sequence to max_seq_length with zeros
        if len(sequence) < self.data_seq_length:
            sequence = np.concatenate((sequence, np.zeros((self.data_seq_length - len(sequence), self.num_channels))))

        # add the query place holder to the end of the sequence
        sequence = np.concatenate((sequence, np.zeros((self.num_queries, self.num_channels))))
        # add one all zero column to the start 
        sequence = np.concatenate((np.zeros((1, sequence.shape[1])), sequence), axis=0)

        segment['data_sequence'] = torch.from_numpy(sequence).type(torch.float32)
        return segment

    def decode_image_from_data(self, data, width, height, num_queries, img_channels=3):
        data = (data.reshape(-1, 5))#.astype(np.uint8)
        h, w = 0, 0
        segment_data = []
        is_data_seq = []
        shape_encoding_seq = []

        for row in data:
            rgb = row[:3] * 255
            shape_encoding = row[3]
            is_data = row[4]

            segment_data.append(rgb.numpy())
            shape_encoding_seq.append(shape_encoding)
            is_data_seq.append(is_data)

            if shape_encoding:
                h += 1
                w = max(w, len(segment_data) // h)

        segment_data = np.array(segment_data)[1:width*height+1, :]
        segment = segment_data.reshape(height, width, 3).astype(np.uint8)
        segment = transforms.ToPILImage()(segment)

        return segment, np.array(is_data_seq), np.array(shape_encoding_seq)
    

if __name__=='__main__':
    pass