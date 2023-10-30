
import numpy as np
import random
import math

import torch
import torchvision
import torchvision.transforms as transforms


class SeqImgClsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, img_size=32, num_queries=64):
        self.dataset = dataset
        self.img_size = img_size
        self.data_seq_length = img_size * img_size
        self.num_queries = num_queries
        self.model_seq_length = self.data_seq_length + self.num_queries + 1

        self.postional_embedding = self.get_positional_embedding()

        self.num_channels = 3 + 1 + 1
        self.channel_info = {
            'RGB': [0,2],
            'postional_embedding': 3,
            'is_data': 4,
        }


    def get_positional_embedding(self):
        x = torch.arange(self.img_size).float()
        y = torch.arange(self.img_size).float()
        x, y = torch.meshgrid(x, y)

        pos_emb = torch.sin(2 * math.pi * x) * torch.sin(2 * math.pi * y)
        max = pos_emb.max()
        min = pos_emb.min()
        return (pos_emb - min) / (max - min)
    

    def add_positional_embedding(self, image):
        pos_emb = self.postional_embedding[:image.shape[1], :image.shape[2]]
        image = torch.cat((image, pos_emb.unsqueeze(0)), dim=0)
        return image


    def preprocess_to_sequence(self, image):
        # flatten the image 
        data = torch.flatten(image, start_dim=1) # -> [channels, height*width=seq_length]
        num_channels, seq_length = data.shape

        # add paddings to the end of the sequence 
        data = torch.cat((data, torch.zeros((num_channels, self.data_seq_length - seq_length))), dim=1) # -> [channels, data_seq_length]

        # add queries placeholder to the end of the sequence 
        data = torch.cat((data, torch.zeros((num_channels, self.num_queries))), dim=1) # -> [channels, data_seq_length + num_queries]

        # add one row (is_data) of zeros to the image 
        data = torch.cat((data, torch.zeros((1, self.data_seq_length + self.num_queries))), dim=0) # -> [channels+1, data_seq_length + num_queries]
        data[-1,:seq_length] = 1

        # add one all zero column to the start 
        data = torch.cat((torch.zeros((data.shape[0], 1)), data), dim=1) # -> [channels+1, data_seq_length+num_queries+1]

        return data.T
    

    def __getitem__(self, index):
        image, label = self.dataset[index]
        # print(f'\nOriginal image: {image.shape}\n{image}')
        
        width = np.random.randint(self.img_size/2, self.img_size)
        height = np.random.randint(self.img_size/2, self.img_size)
        image = transforms.Resize((width, height), antialias=True)(image)
        # print(f'\nResized image: {image.shape}\n{image}')

        image = self.add_positional_embedding(image)
        # print(f'\nAfter adding positional embedding: {image.shape}\n{image}')

        data = self.preprocess_to_sequence(image)
        # print(f'\nAfter preprocessing: {data.shape}\n{data}')

        image_info = {
            'width': width,
            'height': height,
            'label': label,
        }

        return data, image_info

    def __len__(self):
        return len(self.dataset)


def decode_image_from_data(data, width, height, num_queries):
    # reverse the `preprocess_to_sequence` function to get image

    data = data.T
    
    # remove the last row (is_data)
    is_data = data[-1, :]
    data = data[:-1, :]

    # remove the added positional_embedding channel
    positional_embedding = data[-1, 1:width*height+1]
    positional_embedding = positional_embedding.reshape((width, height))
    data = data[:-1, :]

    # remove the first all zero <s> column 
    data = data[:, 1:]

    # remove the queries
    data = data[:, :-num_queries]

    # remove the paddings
    data = data[:, :width*height]

    # reshape the data into [channels, height, width]
    data = data.reshape((3, width, height))

    # convert to PIL image
    image = transforms.ToPILImage()(data)

    return image, is_data, positional_embedding


if __name__ == '__main__':
    # torch.set_printoptions(edgeitems=10)
    
    # trainset = torchvision.datasets.CIFAR100(root='data/cache', train=True, download=True, transform=torchvision.transforms.ToTensor())
    trainset = torchvision.datasets.STL10(root='data/cache', split='train', download=True, transform=torchvision.transforms.ToTensor())
    
    train_dataset = SeqImgClsDataset(
        dataset=trainset,
        img_size=92,
        num_queries=64,
    )

    for i in range(1):
        data, image_info = train_dataset[random.randint(0, len(train_dataset)-1)]
        print(f'\nData shape: {data.shape}\n{data}')

        image, is_data, positional_embedding = decode_image_from_data(data, image_info['width'], image_info['height'], train_dataset.num_queries)

        # save reconstructed image
        image.save(f'reconstructed_{i}.png')




    