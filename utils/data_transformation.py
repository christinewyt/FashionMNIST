import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


class DatasetFromNumpy(torch.utils.data.Dataset):
    '''
    Class to read dataset from numpy format
    
    Args:
        images: images dataset of numpy format
        labels: target labels for dataset
        transform: transform function to be applied on image dataset
    Returns:
        torch dataset type object
    '''
    def __init__(self, images, labels=None, transform=None):
        self.data = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = np.asarray(img)
        
        if self.transform:
            img = self.transform(img)
            
        if self.labels is not None:
            label = self.labels[idx].astype(np.int64)
            label = torch.tensor(label)
            return (img, label)
        else:
            return img


def data_transforms(data):
    '''
    Function which returns the data transformations to be applied on training and testing dataset respectively
    
    Args:
        empty arguments
    Returns:
        transform_train ,transform_test: transformation function for train and test respectively
    '''
    if data == 'mnist' or data == 'fashionmnist':
        trans_img = transforms.ToTensor()
        return trans_img ,trans_img
    
    else:
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        
        return transform_train ,transform_test
    
    

    