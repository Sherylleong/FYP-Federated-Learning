'''
celebreal (real) and celebsynthesis (fake) datasets from kaggle
random subset for train val test split 70: 30: 10
sampling rate 1 fps
To overcome the data imbalance, data augmentation is employed to balance the real and synthetic data. 

50 comm rounds, 10/20 clients
'''
import torch
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from torchvision import models, transforms
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms.v2 as transforms_v2
import torch.nn as nn
import timm

def train_model():
    