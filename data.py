import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
# import cv2
import random


# Define the transforms to be applied to the data
transform = transforms.Compose([transforms.ToTensor()])

# Define a custom dataset class
class SeqImgClsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_seq_length=1024, num_queries=64):
        self.dataset = dataset
        self.max_seq_length = max_seq_length
        self.num_queries = num_queries
        self.num_channels = 3 + 1 + 1  # RGB + `\n` + is_data
        self.index_mapping = np.random.permutation(len(self.dataset))

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

        width = np.random.randint(16, 32)
        height = np.random.randint(16, 32)
        # width, height = 32, 32
        image = self.resize(image, width, height)

        data = self.preprocess(image)
        label_dict = {
            'class': label,
            'width': width,
            'height': height,
        }
        
        # data = torch.ones_like(data)

        # if random.random() < 0.5:
        #     for i in range(data.shape[0]):
        #         data[i, :] = i/data.shape[0]

        # if random.random() < 0.5:
        #     for i in range(data.shape[1]):
        #         data[:, i] *= (i+1)/data.shape[1]

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

    return image, is_data


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


    