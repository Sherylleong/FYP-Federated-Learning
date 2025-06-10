'''
5 clients (random subset)
weighted fedavg early stopping/100
14 comm round
lr 0.01
bs 64
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

    
torch.manual_seed(0)
IMG_SIZE = 224 # change according to model. 224 for effnet, 299 for xception, 224 for resnet
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
BATCH_SIZE = 64
device = 'cuda'
EPOCHS = 100
LR = 0.01
PATIENCE = 5
loss_fn = torch.nn.BCEWithLogitsLoss()

MODEL_NAME = r'ResNet'

# augmentation
class ImageTransform:
    def __init__(self, size, mean, std, train=False):
        if train:
            self.data_transform = transforms.Compose([
                transforms_v2.Resize((size, size), interpolation=Image.BILINEAR),
                transforms_v2.ToTensor(),
                transforms_v2.RandomHorizontalFlip(p=0.5),  # 50% chance to flip
                transforms.Normalize(mean, std)
            ])
        else:  # validation/test Transform (no augmentations)
            self.data_transform = transforms.Compose([
                transforms.Resize((size, size), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    def __call__(self, img):
        return self.data_transform(img)
    
# MODEL CLASSES
class EffNetModel(nn.Module):
    def __init__(self, model_name='efficientnet-b0', num_classes=1):
        super().__init__()
        self.model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model._fc.parameters():
            param.requires_grad = True
    def forward(self, x):
        return self.model(x)
    
class XceptionNetModel(nn.Module):
    def __init__(self, model_name='xception', num_classes=1, device='cuda'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.get_classifier().parameters():
            param.requires_grad = True
        self.model = self.model.to(device)
    def forward(self, x):
        return self.model(x)

class ResNetModel(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=1, device='cuda'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.get_classifier().parameters():
            param.requires_grad = True
        self.model = self.model.to(device)
    def forward(self, x):
        return self.model(x)

class EarlyStopper:
    def __init__(self, patience=PATIENCE, verbose=True):
        self.patience = patience
        self.counter = 0
        self.best_score = float('inf')
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
    


# SPLIT DATASET INTO 5 CLIENTS
from torch.utils.data import random_split
from torchvision import datasets, transforms

def split_dataset_into_clients(combined_dataset, num_clients=5):
    length = len(combined_dataset) # number of data
    # lengths is number of data in each client
    lengths = [length // num_clients] * num_clients # min length of a client dataset
    # distribute remainder data
    for i in range(length % num_clients):
        lengths[i] += 1

    client_datasets = random_split(combined_dataset, lengths, generator=torch.Generator().manual_seed(0))

    client_loaders = [
        DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
        for ds in client_datasets
    ]
    return client_loaders, lengths




import copy

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device).float(), labels.squeeze(0).to(device).float()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy

import os

def save_model(model, model_name=MODEL_NAME, folder=r"lit_review\tinyb4_cctv\models"):
    os.makedirs(folder, exist_ok=True)  # Creates folder if it doesn't exist (relative to current dir)
    save_path = os.path.join(folder, model_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def federated_training(client_train_loaders, client_val_loaders, test_loader, total_train_datasize, comm_rounds=14):
    # change according to model architecture
    global_model = ResNetModel()
    num_clients = len(client_train_loaders)
    print(num_clients)

    for rnd in range(comm_rounds):
        print(f"\n--- Round {rnd+1} ---")
        best_local_models = [{} for _ in range(num_clients)]  # store best model per client

        # train each local model until 100 epochs or early stopping
        for i in range(num_clients):
            client_train_loader = client_train_loaders[i]
            client_val_loader = client_val_loaders[i]

            # initialise local model
            local_model = copy.deepcopy(global_model).to(device)
            client_earlystopper = EarlyStopper()
            
            local_optimizer = torch.optim.Adam(local_model.parameters(), lr=LR)
            criterion = nn.BCEWithLogitsLoss()
            
            best_state = None

            # 100 epochs or until early stopping for local
            for epoch in range(EPOCHS):  
                # train local model
                local_model.train()
                for inputs, labels in client_train_loader:
                    inputs, labels = inputs.to(device), labels.squeeze(0).to(device).float()

                    local_optimizer.zero_grad()

                    outputs = local_model(inputs).squeeze()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    local_optimizer.step()
                
                # validate local model
                local_model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for val_inputs, val_labels in client_val_loader:
                        val_inputs = val_inputs.to(device)
                        val_labels = val_labels.squeeze(0).to(device).float()

                        val_outputs = local_model(val_inputs).squeeze()
                        loss = criterion(val_outputs, val_labels)
                        val_loss += loss.item() * val_inputs.size(0)

                        # compute accuracy
                        probs = torch.sigmoid(val_outputs)           
                        preds = (probs >= 0.5).float()                  
                        correct += (preds == val_labels).sum().item() 
                        total += val_labels.size(0)

                val_loss /= len(client_val_loader.dataset)
                accuracy = correct / total
                print(f"Client {i} Epoch {epoch+1}, Validation Loss: {val_loss:.4f} | Accuracy: {accuracy:.4f}")

                if client_earlystopper.best_val(val_loss):
                    best_state = copy.deepcopy(local_model.state_dict())
                if client_earlystopper.early_stop():
                    break
            if best_state is None:
                best_state = copy.deepcopy(local_model.state_dict())
            best_local_models[i] = best_state
            # save best local model i for comm round
            save_model(best_state, model_name=f'{MODEL_NAME}_{i}_local_round{rnd+1}.pt')


        # aggregate into global model
        with torch.no_grad():
            global_dict = copy.deepcopy(global_model.state_dict())
            for key in global_dict:
                global_dict[key] = sum(
                    (best_local_models[i][key] * (train_client_datasizes[i] / total_train_datasize) for i in range(num_clients))
                )
            global_model.load_state_dict(global_dict)
        save_model(global_model.state_dict(), model_name=f'{MODEL_NAME}_round{rnd+1}.pt')
        print(f"Saved global model to: {f'{MODEL_NAME}_round{rnd+1}.pt'}")

        # test global model
        # can comment out if want to save the training time
        test_loss, test_acc = evaluate_model(global_model, test_loader, criterion)
        print(f'global model: comm round {rnd} - Test Accuracy {test_acc}')

        
if __name__ == "__main__":
    # LOAD DATASETS
    train_transform = ImageTransform(IMG_SIZE, mean, std, train=True)
    val_test_transform = ImageTransform(IMG_SIZE, mean, std, train=False)

    combined_train_dataset = datasets.ImageFolder(r'D:\FF\crops\combined_imagefolder_701515\train', transform=train_transform)
    combined_val_dataset = datasets.ImageFolder(r'D:\FF\crops\combined_imagefolder_701515\val', transform=val_test_transform)
    test_dataset = datasets.ImageFolder(r'D:\FF\crops\combined_imagefolder_701515\test', transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    # SPLIT CLIENTS
    client_train_loaders, train_client_datasizes = split_dataset_into_clients(combined_train_dataset)
    client_val_loaders, val_client_datasizes = split_dataset_into_clients(combined_val_dataset)
    total_train_datasize = sum(train_client_datasizes)
    print(train_client_datasizes)
    print(total_train_datasize)

    # TRAIN AND EVALUATE
    federated_training(client_train_loaders, client_val_loaders, test_loader, total_train_datasize, comm_rounds=14)