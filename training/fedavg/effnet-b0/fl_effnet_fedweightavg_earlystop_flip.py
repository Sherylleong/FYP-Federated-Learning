import os
import numpy as np
import pandas as pd
import matplotlib
import torch
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet

#from albumentations.pytorch import ToTensor 
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader, ConcatDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import sklearn

import cv2
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from torchvision import models, transforms
from torch.utils.data import WeightedRandomSampler
 
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as transforms_v2
deepfake_types = ['Deepfakes', 'Face2Face', 'FaceShifter','FaceSwap', 'NeuralTextures' ]
IMG_SIZE = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
BATCH_SIZE = 64
device = 'cuda'
EPOCHS = 100
LR = 0.001
loss_fn = torch.nn.BCEWithLogitsLoss()
class ImageTransform:
    def __init__(self, size, mean, std, train=False):
        if train:
            self.data_transform = transforms.Compose([
                transforms_v2.Resize((size, size), interpolation=Image.BILINEAR),
                transforms_v2.ToTensor(),
                # transforms.RandomResizedCrop(size, scale=(0.95, 1.0), interpolation=Image.BILINEAR),
                # transforms.RandomResizedCrop(size, scale=(0.8, 1.0), interpolation=Image.BILINEAR),  # Random cropping
                transforms_v2.RandomHorizontalFlip(p=0.5),  # 50% chance to flip
                #transforms.RandomRotation(degrees=5),  # Random rotation
                #transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
                #transforms.GaussianBlur(3),
                # transforms_v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),  # Reduce jitter range
                #transforms_v2.GaussianNoise(mean=0.0, sigma=0.05),
                
                transforms.Normalize(mean, std)
            ])
        else:  # Validation/Test Transform (no augmentations)
            self.data_transform = transforms.Compose([
                transforms.Resize((size, size), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    def __call__(self, img):
        return self.data_transform(img)
# load and prep data

def get_dataset_size(DEEPFAKE_TYPE):
    transformer = ImageTransform(IMG_SIZE, mean, std)

    TRAIN_FOLDER = fr'D:\FF\crops\{DEEPFAKE_TYPE}\train' 
    VAL_FOLDER = fr'D:\FF\crops\{DEEPFAKE_TYPE}\val' 
    TEST_FOLDER = fr'D:\FF\crops\{DEEPFAKE_TYPE}\test' 

    train_ds = ImageFolder(root=TRAIN_FOLDER, transform=transformer)

    return len(train_ds)

def load_dataset(DEEPFAKE_TYPE):
    transformer = ImageTransform(IMG_SIZE, mean, std)
    train_transformer = ImageTransform(IMG_SIZE, mean, std, train=True)

    TRAIN_FOLDER = fr'D:\FF\crops\{DEEPFAKE_TYPE}\train' 
    VAL_FOLDER = fr'D:\FF\crops\{DEEPFAKE_TYPE}\val' 
    TEST_FOLDER = fr'D:\FF\crops\{DEEPFAKE_TYPE}\test' 

    train_ds = ImageFolder(root=TRAIN_FOLDER, transform=train_transformer)
    val_ds = ImageFolder(root=VAL_FOLDER, transform=transformer)
    test_ds = ImageFolder(root=TEST_FOLDER, transform=transformer)

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=64, num_workers=8)
    val_loader = DataLoader(val_ds, shuffle=True, batch_size=64, num_workers=8)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=64, num_workers=8)
    return train_loader, val_loader, test_loader

def load_datasets(deepfake_types):
    deepfake_datasets = {}
    for deepfake_type in deepfake_types:
        train_loader, val_loader, test_loader = load_dataset(deepfake_type)
        deepfake_datasets[deepfake_type] = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    return deepfake_datasets




# clients
def create_clients(deepfake_datasets, initial='clients'): 
    #create a list of client names
    client_names = ['{}_{}'.format(initial, deepfake_type) for deepfake_type in deepfake_types]
    #shard data and place at each client. each shard is a different deepfake type
    shards = deepfake_datasets

    #number of clients must equal number of shards
    assert(len(shards) == len(client_names))

    return {'{}_{}'.format(initial, deepfake_type) : shards[deepfake_type] for deepfake_type in deepfake_types}

def get_client_datasizes(deepfake_types, initial='client'):
    return {'{}_{}'.format(initial, deepfake_type) : get_dataset_size(deepfake_type) for deepfake_type in deepfake_types}



# test model
def test_model(model, test_loader, loss_fn):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = loss_fn

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Model forward pass
            outputs = model(inputs)
            
            # Ensure outputs and labels have the same shape
            outputs = outputs.view(-1)  # Flatten outputs to match labels
            labels = labels.view(-1).float()  # Flatten labels and convert to float

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Calculate predictions and accuracy
            predictions = (torch.sigmoid(outputs) > 0.5).long()
            correct += (predictions == labels.long()).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    print('acc: {:.3%} | loss: {}'.format(accuracy, avg_loss))
    return avg_loss, accuracy


def load_full_dataset():
    transformer = ImageTransform(IMG_SIZE, mean, std)
    train_datasets = []
    val_datasets = []
    test_datasets = []
    for deepfake_type in deepfake_types:
        TRAIN_FOLDER = fr'D:\FF\crops\{deepfake_type}\train' 
        VAL_FOLDER = fr'D:\FF\crops\{deepfake_type}\val' 
        TEST_FOLDER = fr'D:\FF\crops\{deepfake_type}\test' 

        train_ds = ImageFolder(root=TRAIN_FOLDER, transform=transformer)
        val_ds = ImageFolder(root=VAL_FOLDER, transform=transformer)
        test_ds = ImageFolder(root=TEST_FOLDER, transform=transformer)

        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
        test_datasets.append(test_ds)
    full_train_ds = ConcatDataset(train_datasets)
    full_val_ds = ConcatDataset(val_datasets)
    full_test_ds = ConcatDataset(test_datasets)
    train_loader = DataLoader(full_train_ds, shuffle=True, batch_size=64, num_workers=8)
    val_loader = DataLoader(full_val_ds, shuffle=True, batch_size=64, num_workers=8)
    test_loader = DataLoader(full_test_ds, shuffle=False, batch_size=64, num_workers=8)
    return train_loader, val_loader, test_loader


PATIENCE = 5
MODEL_NAME = f'ff_test_fl_effnet_fedweightedavg_earlystop_flip'

def save_checkpoint(model, epoch, loss_history, acc_history, filename=MODEL_NAME):
    checkpoint = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "loss_history": loss_history,
        "acc_history": acc_history,
    }
    filepath = f"{filename}.pth"
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

class EarlyStopper:
    def __init__(self, patience=PATIENCE, verbose=True):
        self.patience = patience
        self.counter = 0
        self.best_score = float('inf')  # Initialize to positive infinity
        self.verbose = verbose
    def best_val(self, val_loss):
        if val_loss < self.best_score:
            self.best_score = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            return False
    def early_stop(self):
        if self.counter >= self.patience:
            if self.verbose:
                print("Early stopping...")
            return True
        return False

# train
def train(earlystopper, epochs, optimizer, model, train_loader, val_loader):
    torch.cuda.empty_cache()
    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []
    for epoch in range(epochs): 
        running_loss = 0
        correct = 0
        total = 0
        model.train()
        for data, labels in train_loader:
            data = data.squeeze(0)
            labels = labels.squeeze(0)
            labels = labels.to(device).float()
            data = data.to(device)
            optimizer.zero_grad()  # clear previous gradients

            #data = data.to(device).float()
            outputs = model(data).squeeze()
            losses = loss_fn(outputs, labels)
            running_loss += losses.item()  # accumulate loss
            
            losses.backward()
            optimizer.step()
            predicted   = (torch.sigmoid(outputs) >= 0.5).float() # calculate if label is 0 or 1
            correct += (predicted == labels).sum().item() 
            total += labels.size(0)
            train_loss_history.append(losses.item())

        average_train_loss = running_loss / total
        average_train_accuracy = correct / total
        train_loss_history.append(average_train_loss)
        train_accuracy_history.append(average_train_accuracy)

        model.eval()
        running_loss = 0
        correct = 0
        total = 0 
        for data, labels in val_loader:
            data = data.squeeze(0)
            labels = labels.squeeze(0)
            labels = labels.to(device).float()
            data = data.to(device).float()

            outputs = model(data).squeeze()
            losses = loss_fn(outputs, labels)
            running_loss += losses.item()  # accumulate loss
            predicted = (torch.sigmoid(outputs) >= 0.5).float() # calculate if label is 0 or 1
            correct += (predicted == labels).sum().item() 
            total += labels.size(0)
        average_val_loss = running_loss / total
        average_val_accuracy = correct / total
        val_loss_history.append(average_val_loss)
        val_accuracy_history.append(average_val_accuracy)
        if earlystopper.best_val(average_val_loss):
            best_model = copy.deepcopy(model)
        if earlystopper.early_stop():
            break
        print(f"Epoch [{epoch}/{epochs}], Train Loss: {average_train_loss:.5f}, Train Accuracy: {average_train_accuracy:.5f}, Val Loss: {average_val_loss:.5f}, Val Accuracy: {average_val_accuracy:.5f}")


    return best_model, train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history

import copy



if __name__ == '__main__':
    device = 'cuda'
    comms_round = 10
    lr = 0.01
    loss=torch.nn.BCEWithLogitsLoss()
    global_earlystopper = EarlyStopper()

    global_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1)
    for param in global_model.parameters():
        param.requires_grad = False
    # Unfreeze the final fully connected layer
    for param in global_model._fc.parameters():
        param.requires_grad = True
    global_model = global_model.to(device)

    train_batched, _, test_batched = load_full_dataset()
    train_batched_size = len(train_batched)

    average_weights=copy.deepcopy(global_model.state_dict())



    deepfake_datasets = load_datasets(deepfake_types)
    clients = create_clients(deepfake_datasets, initial='client')
    client_datasizes = get_client_datasizes(deepfake_datasets)
    client_datasizes['global'] = sum(client_datasizes.values())
    print(client_datasizes)

    global_loss_histories = []
    global_acc_histories = []
    for comm_round in range(comms_round):  
        print(comm_round, 'comm')
        scaled_local_weight_list = list()

        clients_batched = clients
        client_names= list(clients_batched.keys())

        for client in client_names:
            print(client)
            local_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1)
            for param in local_model.parameters():
                param.requires_grad = False
            # Unfreeze the final fully connected layer
            for param in local_model._fc.parameters():
                param.requires_grad = True
            global_weights = copy.deepcopy(global_model.state_dict())
            local_model.load_state_dict(global_weights)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=LR)
            local_model = local_model.to(device)
            train_loader, val_loader, test_loader = clients_batched[client]['train'], clients_batched[client]['val'], clients_batched[client]['test']
            earlystopper = EarlyStopper()
            local_model, train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history = train(earlystopper, 100, optimizer, local_model, train_loader, val_loader)
            weights=copy.deepcopy(local_model.state_dict())

        average_weights = {key: 0 for key in weights}
        for key in weights:
            average_weights[key] += weights[key]
        average_weights[key] = average_weights[key] * client_datasizes[client] / client_datasizes['global']

        #clear session to free memory after each communication round
        torch.cuda.empty_cache()


        #update global model
        global_model_weights = copy.deepcopy(average_weights)
        global_model.load_state_dict(global_model_weights)

        #test global model and print out metrics after each communications round
        print('global')
        global_loss, global_acc = test_model(global_model, test_loader, loss_fn)
        if global_earlystopper.best_val(global_acc):
            save_checkpoint(global_model, comm_round, global_loss_histories, global_acc_histories)
        if global_earlystopper.early_stop():
            break
        global_loss_histories.append(global_loss)
        global_acc_histories.append(global_acc)
        save_checkpoint(global_model, comm_round, global_loss_histories, global_acc_histories)