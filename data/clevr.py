import os
import torch
import json
import tqdm
from PIL import Image


class CLEVR(torch.utils.data.Dataset):

    def __init__(
            self, dataset_path, only_return_img_path=False, 
            split='train', sample_truncation=-1,
            ):

        self.only_return_img_path = only_return_img_path
        annotation_file = os.path.join(dataset_path, 'captions', f'{split}.json')
        self.samples = json.load(open(annotation_file, 'r'))
        if sample_truncation > 0:
            self.samples = self.samples[:sample_truncation]
        self.dataset_path = dataset_path


    def validate_exist(self, valid_img_paths=None):
        validated_samples = []
        not_exist = []
        not_exist_in_provided_list = []
        for i in tqdm.tqdm(range(len(self.samples))):
            try:
                self.__getitem__(i)
            except:
                not_exist.append(self.samples[i])
                continue
            
            if valid_img_paths is not None:
                if self.__getitem__(i, only_return_img_path=True) not in valid_img_paths:
                    not_exist_in_provided_list.append(self.samples[i])
                else:
                    validated_samples.append(self.samples[i])
            else:
                validated_samples.append(self.samples[i])

        self.samples = validated_samples
            
        print(f'Found {len(not_exist)} samples failed to load due to file not exist.')
        if len(not_exist_in_provided_list) > 0:
            print(f'Found {len(not_exist_in_provided_list)} samples failed to load due to file not exist in provided list.')
        print(f'After validation, {len(self.samples)} samples left.')

        return not_exist


    def process_clevr_sample(self, img_path, caption, start_path='<img_path>', end_path='</img_path>', human_turn='### Human: \n', gpt_turn='### AI: \n', eos_token='<|endoftext|>'):
        result = f'{human_turn}Describe this image<img_path>{img_path}</img_path>{eos_token}\n'
        result += f'{gpt_turn}{caption}{eos_token}'
        return result

    def __getitem__(self, index, only_return_img_path=False):
        sample = self.samples[index]
        img_path = os.path.join(self.dataset_path, sample['img_path'])
        caption = sample['caption']

        # img = Image.open(img_path)
        assert os.path.exists(img_path), f'Image not found: {img_path}'

        if self.only_return_img_path or only_return_img_path:
            return img_path
        else:
            return self.process_clevr_sample(img_path, caption)
    
    def __len__(self):
        return len(self.samples)