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
try:
    from lvis import LVIS
except:
    pass # temporary fix for the lvis & opencv installation issue
import random


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

        self.num_segments = 200000000
        self.num_images = len(self.img_ids)
        self.num_categories = 1

    def preprocess_annotations(self, annotations, min_pixel_num):
        new_annotations = []
        for annotation in annotations:
            mask = decode(annotation['segmentation'])
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
    
    def load_segments_from_one_image(self, image_index=None, min_pixel_num=16):
        if image_index is None:
            success = False
            while not success:
                image_index = np.random.randint(0, len(self.img_ids))
                img_path = self.img_ids[image_index] + '.jpg'
                json_path = self.img_ids[image_index] + '.json'
                if os.path.exists(json_path):
                    try:
                        image = np.array(Image.open(img_path).convert('RGB'))
                        annotations = json.load(open(json_path))['annotations']
                        annotations = self.preprocess_annotations(annotations, min_pixel_num)
                        success = True
                    except:
                        pass

        segments = []
        for annotation in annotations:
            mask = annotation['segmentation']
            bbox = [int(b) for b in annotation['bbox']]
            bbox[2] = 1 if bbox[2] == 0 else bbox[2]
            bbox[3] = 1 if bbox[3] == 0 else bbox[3]
            
            masked_img = image * mask[:, :, None]
            x, y, w, h = bbox
            patch = masked_img[y:y+h, x:x+w]
            patch_mask = mask[y:y+h, x:x+w]
            segments.append({
                "patch": patch,
                "mask": patch_mask,
                'image_path': img_path,
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
        self.num_images = 108077
        self.num_segments = 0
        for line in self.file:
            self.num_segments += len(json.loads(line)['objects'])
        self.file.seek(0)
        self.num_categories = 1

    def load_segments_from_one_image(self):
        line = self.file.readline()
        if not line:
            self.file.seek(0)
            line = self.file.readline()
        line = json.loads(line)

        image_path = os.path.join(self.visual_genome_root, line['image'])
        image = np.array(Image.open(image_path).convert('RGB'))
        objects = line['objects']
        segments = []
        for obj in objects:
            mask = decode(obj['mask'])
            bbox = [int(b) for b in obj['bbox']]
            bbox[2] = 1 if bbox[2] == 0 else bbox[2]
            bbox[3] = 1 if bbox[3] == 0 else bbox[3]
            
            masked_img = image * mask[:, :, None]
            x, y, w, h = bbox
            patch = masked_img[y:y+h, x:x+w]
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
        self.file = open(self.annotations_path, 'r')
        
        self.category_info = json.load(open(os.path.join(v3det_root, 'annotations/v3det_2023_v1_category_info.json')))
        self.num_categories = len(self.category_info)

        self.num_images = 0
        self.num_segments = 0
        for line in self.file:
            self.num_segments += len(json.loads(line)['objects'])
            self.num_images += 1
        self.file.seek(0)

    def load_segments_from_one_image(self):
        line = self.file.readline()
        if not line:
            self.file.seek(0)
            line = self.file.readline()
        line = json.loads(line)

        image_path = os.path.join(self.images_path, line['image'])
        image = np.array(Image.open(image_path).convert('RGB'))
        objects = line['objects']
        segments = []
        for obj in objects:
            mask = decode(obj['mask'])
            bbox = [int(b) for b in obj['bbox']]
            bbox[2] = 1 if bbox[2] == 0 else bbox[2]
            bbox[3] = 1 if bbox[3] == 0 else bbox[3]
            
            masked_img = image * mask[:, :, None]
            x, y, w, h = bbox
            patch = masked_img[y:y+h, x:x+w]
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
        
    def preprocess_annotations(self, annotations, min_pixel_num):
        new_annotations = []
        for annotation in annotations:
            mask = self.lvis.ann_to_mask(annotation)
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

    def load_segments_from_one_image(self, image_index=None, min_pixel_num=16):
        if image_index is None:
            img_id = random.choice(self.img_ids)

        img = self.lvis.load_imgs([img_id])[0]
        img_path = img['coco_url'].replace('http://images.cocodataset.org', os.path.join(self.coco_root, 'images'))
        image = np.array(Image.open(img_path).convert('RGB'))

        annotations = self.lvis.load_anns(self.lvis.get_ann_ids([img_id]))
        annotations = self.preprocess_annotations(annotations, min_pixel_num)
        segments = []
        for annotation in annotations:
            label = annotation['category_id']
            mask = annotation['segmentation']
            bbox = [int(b) for b in annotation['bbox']]
            bbox[2] = 1 if bbox[2] == 0 else bbox[2]
            bbox[3] = 1 if bbox[3] == 0 else bbox[3]
            
            masked_img = image * mask[:, :, None]
            x, y, w, h = bbox
            patch = masked_img[y:y+h, x:x+w]
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
    def __init__(self, coco_root, split):
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

    def load_segments_from_one_image(self, image_index=None, min_pixel_num=16):
        if image_index is None:
            image_index = np.random.randint(0, len(self.img_ids))

        img_path = f"{self.coco_root}/images/{self.sub_dir}/{self.coco.loadImgs(self.img_ids[image_index])[0]['file_name']}"
        image = np.array(Image.open(img_path).convert('RGB'))

        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=self.img_ids[image_index]))
        annotations = self.preprocess_annotations(annotations, min_pixel_num)
        segments = []
        for annotation in annotations:
            label = annotation['category_id']
            mask = annotation['segmentation']
            bbox = [int(b) for b in annotation['bbox']]
            bbox[2] = 1 if bbox[2] == 0 else bbox[2]
            bbox[3] = 1 if bbox[3] == 0 else bbox[3]
            
            masked_img = image * mask[:, :, None]
            x, y, w, h = bbox
            patch = masked_img[y:y+h, x:x+w]
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
    def __init__(self, dataset, num_queries, data_seq_length=64*64, min_resize_ratio=1.0):
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

    def __getitem__(self, index):
        while len(self.sample_buffer) == 0:
            segments = self.dataset.load_segments_from_one_image()
            for segment in segments:
                segment = self.resize(segment)
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
        }
        return segment['data_sequence'], segment_info

    def __len__(self):
        return self.virtual_dataset_size

    def update_data_seq_length_multiplier(self, data_seq_length_multiplier):
        self.data_seq_length_multiplier = data_seq_length_multiplier
        

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