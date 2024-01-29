import os
import torch
import json
import tqdm
from PIL import Image


class ShareGPT4V(torch.utils.data.Dataset):

    def __init__(self, annotation_file, dir_mapping=None, only_return_img_path=False):

        if dir_mapping is None:
            dir_mapping = {
                'sam/images': '/home/dchenbs/workspace/datasets/sa1b',
                'coco/train2017': '/home/dchenbs/workspace/datasets/coco2017/images/train2017',
                'llava/llava_pretrain/images': '/home/dchenbs/workspace/datasets/sharegpt4v/LLaVA-Pretrain',
                'wikiart/images': '/home/dchenbs/workspace/datasets/sharegpt4v/WebData/wikiart/images',
                'share_textvqa/images': '/home/dchenbs/workspace/datasets/sharegpt4v/WebData/share_textvqa/images',
                'web-celebrity/images': '/home/dchenbs/workspace/datasets/sharegpt4v/WebData/web-celebrity/images',
                    'Choi_Min-sik': 'Choi_Min_sik', 
                    'Lee_Byung-hun': 'Lee_Byung_hun',
                'web-landmark/images': '/home/dchenbs/workspace/datasets/sharegpt4v/WebData/web-landmark/images',
                'vg/VG_100K': '/home/dchenbs/workspace/datasets/VisualGenome/VG_100K',
                'vg/VG_100K_2': '/home/dchenbs/workspace/datasets/VisualGenome/VG_100K_2',
                'gqa/images': '/home/dchenbs/workspace/datasets/gqa/images',
                'ocr_vqa/images': '/home/dchenbs/workspace/datasets/sharegpt4v/ocr_vqa/images',
                'textvqa/train_images': '/home/dchenbs/workspace/datasets/sharegpt4v/textvqa/train_images',
            }

        self.dir_mapping = dir_mapping
        self.only_return_img_path = only_return_img_path

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


    def process_sharegpt4v_sample(self, img_path, sample, start_path='<img_path>', end_path='</img_path>', human_turn='### Human: \n', gpt_turn='### AI: \n', eos_token='<|endoftext|>'):
        result = ''
        for utterance in sample['conversations']:
            result += human_turn if utterance['from']=='human' else gpt_turn
            result += utterance['value'].replace('<image>', f'{start_path}{img_path}{end_path}') + eos_token + '\n'
        return result

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

        if self.only_return_img_path:
            return img_path
        else:
            return self.process_sharegpt4v_sample(img_path, sample)
    
    def __len__(self):
        return len(self.samples)