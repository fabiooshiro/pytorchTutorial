import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, dataset
import numpy as np
import math

class WineDataset(Dataset):

    def __init__(self, transform=None):
        # super().__init__()
        self.transform = transform
        xy = np.loadtxt('./data_wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]] # n_samples, 1
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        # len(dataset)
        return self.n_samples

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
        
class MulTransform:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets

# dataset = WineDataset(transform=ToTensor())
dataset = WineDataset(transform=None)
first_data = dataset[0]
features, labels = first_data
print(features)

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]

features, labels = first_data
print(type(features), type(labels))
print(features)