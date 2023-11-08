import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from scipy.ndimage import label



class COCOMaskDataset(Dataset):
    def __init__(self, coco_root, split, num_queries, virtual_dataset_size=1000, data_seq_length=64*64, min_pixel_num=16):
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

        self.sample_buffer = []
        self.virtual_dataset_size = virtual_dataset_size

        self.min_pixel_num = min_pixel_num
        self.num_queries = num_queries
        self.data_seq_length = data_seq_length
        self.model_seq_length = self.data_seq_length + self.num_queries + 1

        self.num_channels = 3 + 1 + 1
        self.channel_info = {
                'data': [0,3],
                'shape_encoding': 3,
                'is_data': 4,
            }

    def load_image(self, image_index=None):
        if image_index is None:
            image_index = np.random.randint(0, len(self.img_ids))

        img_id = self.img_ids[image_index]
        img_path = f"{self.coco_root}/images/{self.sub_dir}/{self.coco.loadImgs(img_id)[0]['file_name']}"
        img = np.array(Image.open(img_path).convert('RGB'))

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        annotations = self.preprocess_annotations(annotations)

        labels = [annotation['category_id'] for annotation in annotations]
        # masks = [self.coco.annToMask(annotation) for annotation in annotations]
        masks = [annotation['segmentation'] for annotation in annotations]
        bboxs = [[int(b) for b in annotation['bbox']] for annotation in annotations]
            
        samples = []
        for mask, bbox, label in zip(masks, bboxs, labels):
            
            masked_img = img * mask[:, :, None]

            x, y, w, h = bbox
            cropped_img = masked_img[y:y+h, x:x+w]
            cropped_mask = mask[y:y+h, x:x+w]
            cropped_img, cropped_mask = self.resize_to_max_seq_length(cropped_img, cropped_mask)

            data_sequence = self.encode_segment_to_sequence(cropped_img, cropped_mask)
            # decoded_segment, decoded_mask = self.decode_sequence_to_segment(data_sequence)

            samples.append({
                "data_sequence": torch.from_numpy(data_sequence),
                'width': w,
                'height': h,
                "label": label,
                # "path": img_path,
                # "segment": cropped_img,
                # "mask": cropped_mask,
                # "decoded_segment": decoded_segment,
                # "decoded_mask": decoded_mask,
            })  
        self.sample_buffer.extend(samples)

    def resize_to_max_seq_length(self, segment, mask):
        h, w = mask.shape
        if h * w > self.data_seq_length:
            ratio = np.sqrt(self.data_seq_length / (h * w))
            h = int(h * ratio)
            w = int(w * ratio)
            segment = np.array(Image.fromarray(segment).resize((w, h)))
            mask = np.array(Image.fromarray(mask).resize((w, h)))
        return segment, mask
    
    def preprocess_annotations(self, annotations):
        new_annotations = []
        for annotation in annotations:
            mask = self.coco.annToMask(annotation)
            labeled_mask, num_labels = label(mask)
            for i in range(1, num_labels + 1):
                new_annotation = annotation.copy()
                new_annotation['segmentation'] = (labeled_mask == i)
                if new_annotation['segmentation'].sum() >= self.min_pixel_num:
                    new_annotation['bbox'] = [
                        np.min(np.where(new_annotation['segmentation'])[1]),  # x_min
                        np.min(np.where(new_annotation['segmentation'])[0]),  # y_min
                        np.max(np.where(new_annotation['segmentation'])[1]) - np.min(np.where(new_annotation['segmentation'])[1]),  # width
                        np.max(np.where(new_annotation['segmentation'])[0]) - np.min(np.where(new_annotation['segmentation'])[0]),  # height
                    ]
                    new_annotations.append(new_annotation)
            break
        return new_annotations

    def encode_segment_to_sequence(self, segment, mask):
        h, w, _ = segment.shape
        sequence = []

        for i in range(h):
            for j in range(w):
                pixel_data = segment[i, j] / 255.0
                is_rightmost = 1 if j == w - 1 else 0
                is_non_masked = int(mask[i, j])

                sequence.append(pixel_data.tolist() + [is_rightmost, is_non_masked])

        sequence = np.array(sequence) 
        # pad the sequence to max_seq_length with zeros
        sequence = np.concatenate((sequence, np.zeros((self.data_seq_length - len(sequence), 5))))

        # add the query place holder to the end of the sequence
        sequence = np.concatenate((sequence, np.zeros((self.num_queries, 5))))

        # add one all zero column to the start 
        sequence = np.concatenate((np.zeros((1, sequence.shape[1])), sequence), axis=0)
        return sequence

    def decode_sequence_to_segment(self, sequence):
        sequence = sequence[1:-self.num_queries, :]
        sequence = (sequence.reshape(-1, 5))#.astype(np.uint8)
        h, w = 0, 0
        segment_data = []
        mask_data = []

        for row in sequence:
            rgb = row[:3] * 255
            is_rightmost = row[3]
            is_non_masked = row[4]

            segment_data.append(rgb)
            mask_data.append(is_non_masked)

            if is_rightmost:
                h += 1
                w = max(w, len(segment_data) // h)

        # truncate the segment_data and mask according to the last `is_rightmost`
        last_rightmost_index = np.where(sequence[:, 3] == 1)[0][-1]
        segment_data = segment_data[:last_rightmost_index + 1]
        mask_data = mask_data[:last_rightmost_index + 1]
        
        segment = np.array(segment_data).reshape(h, w, 3).astype(np.uint8)
        mask = np.array(mask_data).reshape(h, w).astype(bool)
        return segment, mask

    def __getitem__(self, index):
        while len(self.sample_buffer) == 0:
            self.load_image()
           
        sample = self.sample_buffer.pop() 
        sample_info = {
            'width': sample['width'],
            'height': sample['height'],
            'label': sample['label'],
        }
        return sample['data_sequence'], sample_info

    def __len__(self):
        return self.virtual_dataset_size




if __name__ == '__main__':
    dataset = COCOMaskDataset('/home/dchenbs/workspace/datasets/coco2017', 'val', num_queries=64)

    for i in range(1):
        data_sequence, sample_info = dataset[i]
        print(f"data_sequence: {data_sequence.shape}\n{data_sequence}")


        # # visualize image, segment and mask
        # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # axes[0].imshow(Image.open(sample['path']))
        # axes[1].imshow(sample['segment'] * sample['mask'][:, :, None])
        # axes[2].imshow(sample['mask'])
        # plt.suptitle(f"label: {dataset.id_to_name[sample['label']]}")
        # plt.show()

        
        # # visualize image, segment and mask
        # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # axes[0].imshow(Image.open(sample['path']))
        # axes[1].imshow(sample['decoded_segment'] * sample['decoded_mask'][:, :, None])
        # axes[2].imshow(sample['decoded_mask'])
        # plt.suptitle(f"label: {dataset.id_to_name[sample['label']]}")
        # plt.show()
