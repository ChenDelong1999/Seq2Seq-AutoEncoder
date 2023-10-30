
import numpy as np
import random

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets import CIFAR10, CIFAR100, STL10


# Define the transforms to be applied to the data
transform = transforms.Compose([transforms.ToTensor()])

def get_dataset(args):
    if args.dataset=='cifar10':
        train_dataset = SeqImgClsDataset(
            dataset=CIFAR10(root=args.data_dir, train=True, download=True, transform=None),
            img_size=32,
            num_queries=args.num_queries,
        )
        test_dataset = SeqImgClsDataset(
            dataset=CIFAR10(root=args.data_dir, train=False, download=True, transform=None),
            img_size=32,
            num_queries=args.num_queries,
        )
    elif args.dataset=='cifar100':
        train_dataset = SeqImgClsDataset(
            dataset=CIFAR100(root=args.data_dir, train=True, download=True, transform=None),
            img_size=32,
            num_queries=args.num_queries,
        )
        test_dataset = SeqImgClsDataset(
            dataset=CIFAR100(root=args.data_dir, train=False, download=True, transform=None),
            img_size=32,
            num_queries=args.num_queries,
        )
    elif args.dataset=='stl10':
        train_dataset = SeqImgClsDataset(
            dataset=STL10(root=args.data_dir, split='train+unlabeled', download=True, transform=None),
            img_size=64,
            num_queries=args.num_queries,
        )
        test_dataset = SeqImgClsDataset(
            dataset=STL10(root=args.data_dir, split='test', download=True, transform=None),
            img_size=64,
            num_queries=args.num_queries,
        )
    return train_dataset, test_dataset


# Define a custom dataset class
class SeqImgClsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, img_size=1024, num_queries=64):
        self.dataset = dataset
        self.img_size = img_size
        self.max_seq_length = img_size * img_size
        self.num_queries = num_queries
        self.num_channels = 3 + 1 + 1  # RGB + `\n` + is_data
        self.index_mapping = np.random.permutation(len(self.dataset)) # shuffle the dataset

    def resize(self, image, width, height):

        image = transforms.Resize((height, width))(image)
        image = transforms.ToTensor()(image)
        # print(f'\nResized image: {image.shape}\n{image}')

        # add another channel to the image
        image = torch.cat((image, torch.zeros((1, height, width))), dim=0)
        
        # the right most column of the added channle is all ones
        image[-1, :, -1] = 1
        # print(f'\nAfter adding another channel: {image.shape}\n{image}')

        return image
    
    def preprocess(self, image):
        # flatten the image into [channels, (height*width=seq_length)]
        data = torch.flatten(image, start_dim=1)
        num_channels, seq_length = data.shape
        # print(f'\nBefore padding: {data.shape}\n{data}')

        # add paddings to the end of the sequence -> [channels, max_seq_length]
        data = torch.cat((data, torch.zeros((num_channels, self.max_seq_length - seq_length))), dim=1)
        # print(f'\nAfter padding: {data.shape}\n{data}')

        # add queries to the end of the sequence -> [channels, max_seq_length+num_queries]
        data = torch.cat((data, torch.zeros((num_channels, self.num_queries))), dim=1)
        # print(f'\nAfter adding queries: {data.shape}\n{data}')

        # add one row (is_data) of zeros to the image -> [channels+2, max_seq_length+num_queries]
        data = torch.cat((data, torch.zeros((1, self.max_seq_length + self.num_queries))), dim=0)
        data[-1,:seq_length] = 1
        # print(f'\nAfter adding is_data : {data.shape}\n{data}')

        # add one all zero column to the start -> [channels+2, max_seq_length+num_queries+1]
        data = torch.cat((torch.zeros((data.shape[0], 1)), data), dim=1)
        # print(f'\nAfter adding all zero column : {data.shape}\n{data}')

        return data.T
    

    def __getitem__(self, index):
        index = self.index_mapping[index]
        image, label = self.dataset[index]

        width = np.random.randint(self.img_size/2, self.img_size)
        height = np.random.randint(self.img_size/2, self.img_size)
        # width, height = 32, 32
        image = self.resize(image, width, height)

        data = self.preprocess(image)
        label_dict = {
            'class': label,
            'width': width,
            'height': height,
        }

        return data, label_dict

    def __len__(self):
        return len(self.dataset)


def decode_image_from_data(data, width, height, num_queries):
    # reverse the `preprocess` function to get image

    data = data.T
    
    # remove the last row (is_data)
    is_data = data[-1, :]
    data = data[:-1, :]

    # remove the added channel (`\n`)
    return_channel = data[-1, :]
    data = data[:-1, :]

    
    # remove the first all zero column
    data = data[:, 1:]


    # remove the queries
    data = data[:, :-num_queries]

    # remove the paddings
    data = data[:, :width*height]

    # reshape the data into [channels, height, width]
    data = data.reshape((3, height, width))

    # convert to PIL image
    image = transforms.ToPILImage()(data)

    return image, is_data, return_channel


if __name__ == '__main__':
    
    trainset = torchvision.datasets.CIFAR100(root='dataset', train=True, download=True, transform=transform)
    
    train_dataset = SeqImgClsDataset(
        dataset=trainset,
        max_seq_length=1024,
        num_queries=64,
    )

    for i in range(1):
        data, label_dict = train_dataset[i]
        print(f'\nData shape: {data.shape}\n{data}')

        # image, is_data = decode_image_from_data(data, label_dict['width'], label_dict['height'], train_dataset.num_queries)

        # # save 
        # image.save(f'{i}-decoded.png')

        # # visualize the original image
        # image, label = trainset[i]
        # image = transforms.ToPILImage()(image)
        # image.save(f'{i}-original.png')


    