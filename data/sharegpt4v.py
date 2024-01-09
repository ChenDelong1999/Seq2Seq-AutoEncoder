import os
import torch
import json
import tqdm
from PIL import Image


class ShareGPT4V(torch.utils.data.Dataset):

    def __init__(self, annotation_file, dir_mapping):

        self.dir_mapping = dir_mapping

        samples = json.load(open(annotation_file, 'r'))
        self.samples = [s for s in samples if 'image' in s]
        print(f'Total samples: {len(samples)}, after removing text-only samples: {len(self.samples)}')

        self.sam_dir_mapping = {}
        for i in range(51):
            files = os.listdir(os.path.join(dir_mapping['sam/images'], f"sa_{i:06}"))
            for file in files:
                if file.endswith('.jpg'):
                    self.sam_dir_mapping[file] = f"sa_{i:06}/{file}"

    def validate_exist(self):
        validated_samples = []
        failed_samples = []
        for i in tqdm.tqdm(range(len(self.samples))):
            try:
                self.__getitem__(i)
            except:
                failed_samples.append(self.samples[i])
                continue
            validated_samples.append(self.samples[i])
            
        print(f'Found {len(failed_samples)} samples failed to load.')
        self.samples = validated_samples
        return failed_samples


    def __getitem__(self, index):
        sample = self.samples[index]
        img_path = sample['image']

        if 'sam/images' in img_path:
            relative_img_path = img_path.split('sam/images/')[-1]
            img_path = img_path.replace(relative_img_path, self.sam_dir_mapping[relative_img_path])

        for org, new in self.dir_mapping.items():
            img_path = img_path.replace(org, new)

        # img = Image.open(img_path)
        assert os.path.exists(img_path), f'Image not found: {img_path}'

        return img_path, sample
    
    def __len__(self):
        return len(self.samples)