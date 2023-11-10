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

class SA1BDataset:
    def __init__(self, sa1b_root):
        self.sa1b_root = sa1b_root
        self.img_ids = []
        
        for subfolder in os.listdir(sa1b_root):
            subfolder_path = os.path.join(sa1b_root, subfolder)
            if os.path.isdir(subfolder_path):
                for img_file in os.listdir(subfolder_path):
                    if img_file.endswith('.jpg'):
                        self.img_ids.append(os.path.join(subfolder_path, img_file[:-4]))
        print(f"SA1B dataset loaded. {len(self.img_ids)} images found.")

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
            break
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
                "label": 0,
                "name": "",
            })  

        return segments

class COCODataset:
    def __init__(self, coco_root, split):
        self.coco_root = coco_root
        if split == 'train':
            self.sub_dir = 'train2017'
        elif split == 'val':
            self.sub_dir = 'val2017'
        else:
            raise NotImplementedError

        self.coco = COCO(f'{self.coco_root}/annotations/instances_{self.sub_dir}.json')
        self.img_ids = list(sorted(self.coco.imgs.keys()))
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
            break
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
            })  

        return segments

class SeqMaskDataset(Dataset):
    def __init__(self, dataset, num_queries, virtual_dataset_size=1000, data_seq_length=64*64, min_resize_ratio=1.0):
        self.dataset = dataset
        self.virtual_dataset_size = virtual_dataset_size
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
            'label': segment['label'],
            'image_path': segment['image_path'],
            'bbox': segment['bbox'],
        }
        return segment['data_sequence'], segment_info

    def __len__(self):
        return self.virtual_dataset_size

    def resize(self, segment):
        h, w = segment['patch'].shape[:2]
        if h * w > self.data_seq_length:
            ratio_to_maxlength = np.sqrt(self.data_seq_length / (h * w))
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
    from pycocotools.coco import COCO

    # Initialize COCO API for instance annotations
    coco = COCO('/home/dchenbs/workspace/datasets/coco2017/annotations/instances_train2017.json')

    # Get all annotation ids
    annotation_ids = coco.getAnnIds()

    # Print the total number of annotations
    print(f'Total number of instance annotations: {len(annotation_ids)}')

    print(coco.loadAnns(annotation_ids[0]))

    
    # from PIL import Image
    # import matplotlib.pyplot as plt
    # import tqdm


    # sa1b_dataset = SA1BDataset('/home/dchenbs/workspace/datasets/sa1b')
    # seq_mask_dataset = SeqMaskDataset(sa1b_dataset, num_queries=64)

    # # coco_dataset = COCODataset('/home/dchenbs/workspace/datasets/coco2017', 'val')
    # # seq_mask_dataset = SeqMaskDataset(coco_dataset, num_queries=64)

    # for i in tqdm.tqdm(range(500)):
    #     segment, segment_info = seq_mask_dataset[i]

    #     # print(segment.shape)

    #     # segment, is_data_seq, shape_encoding_seq = seq_mask_dataset.decode_image_from_data(
    #     #     segment, 
    #     #     segment_info['width'], 
    #     #     segment_info['height'], 
    #     #     seq_mask_dataset.num_queries, 
    #     #     img_channels=seq_mask_dataset.img_channels
    #     #     )
    #     # plt.figure(figsize=(20, 10))
    #     # plt.subplot(1, 2, 1)
    #     # plt.imshow(Image.open(segment_info['image_path']))
    #     # x, y, w, h = segment_info['bbox']
    #     # rect = plt.Rectangle((x, y), w, h, fill=False, color='red')
    #     # plt.gca().add_patch(rect)

    #     # plt.subplot(1, 2, 2)
    #     # plt.imshow(segment)
        
    #     # plt.savefig(f'test{i}.png')
    #     # plt.close()
