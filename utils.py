import os
import sys
import time
import math
import PIL.Image as Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pandas as pd
import numpy as np

from tqdm import tqdm

class Food_Dataset(Dataset):
    def __init__(self, split): # split: train or test
      
      self.split = split
      self.root_folder = '.'
      if split == 'train' :
        csv = pd.read_csv(f'{self.root_folder}/train_data_info.csv')
        self.image_name_list = csv['image_name'].to_list()
        self.label_list = csv['label'].to_list()
      elif split == 'test' :
        csv = pd.read_csv(f'{self.root_folder}/submission.csv')
        self.image_name_list = csv['image_name'].to_list()
        
      self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
      return len(self.image_name_list)

    def __getitem__(self, idx):

      image_path = f'{self.root_folder}/{self.split}_images/{self.image_name_list[idx]}'

      image = self.to_tensor(Image.open(image_path).convert("RGB"))

      if self.split == 'train' :
        label = torch.Tensor([self.label_list[idx]]).type(torch.LongTensor)
        return image, label

      elif self.split == 'test' :
        return image
      
class Food_Subset_toset(Dataset):
    def __init__(self, subset, transform=None):
      self.subset = subset
      self.transform = transform
    
    def __getitem__(self, index):
      data = self.subset[index]
      if isinstance(data, tuple) and len(data) == 2:
        x, y = data
        if self.transform:
            x = self.transform(x)
        return x, y
      else:
        if self.transform:
            data = self.transform(data)
        return data
        
    def __len__(self):
        return len(self.subset)

def get_mean_std(dataset, batch_size=1, num_workers=1):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         num_workers=num_workers)
    data = next(iter(loader))
    if len(data) == 2:
      x, y = data
      mean = torch.mean(x, dim=(0, 2, 3))  
      std = torch.std(x, dim=(0, 2, 3))  
      return mean, std
    else :
      mean = torch.mean(data, dim=(0, 2, 3))
      std = torch.std(data, dim=(0, 2, 3))
      return mean, std

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
      
class ToFloatTensor(object):
    def __call__(self, pic):
        return pic.clone().detach().to(torch.float32) / 255.0
      
class ToUInt8Tensor(object):
    def __call__(self, pic):
        return (pic.clone().detach() * 255).to(torch.uint8)

def classwise_accuracy(predicted_labels, true_labels, num_classes=27):
    predicted = np.array(predicted_labels)
    true = np.array(true_labels)
    
    class_counts = {i: 0 for i in range(num_classes)}
    class_correct = {i: 0 for i in range(num_classes)}
    class_accuracy = {i: 0 for i in range(num_classes)}
    
    for pred, label in zip(predicted, true):
        class_counts[label] += 1
        if pred == label:
            class_correct[label] += 1
    
    for i in range(num_classes):
        if class_counts[i] != 0:
            class_accuracy[i] = class_correct[i] / class_counts[i]
    
    return class_accuracy, class_counts