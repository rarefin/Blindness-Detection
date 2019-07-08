from .DataLoader import TrainDataset
from .Trainer import train
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
import torchvision
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from torch import nn
import json
from os.path import join
import pandas as pd

# seed all RNGs for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# CuDNN reproducibility options
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# read the config file
with open("config.json", "r") as read_file:
    config = json.load(read_file)


data_dir = config["paths"]["data_dir"]
train_df = pd.read_csv(join(data_dir, 'train.csv'))
train_dir = join(data_dir, 'train_images')


# Split data into train and validation
val_proportion = config["training"]["val_proportion"]
data_indices = range(0, len(train_df))
train_indices, val_indices = train_test_split(data_indices, test_size=val_proportion, random_state=1, shuffle=True)

# Create Dataloader
BATCH_SIZE = config["training"]["batch_size"]
transform = transforms.Compose([
    transforms.Resize(256), # Resize the dataset
    transforms.CenterCrop(224),
    transforms.ToTensor(), # because inputs dtype is PIL Image
])
train_dataset = TrainDataset(train_dir, train_df, train_indices, transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = TrainDataset(train_dir, val_indices, transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

NUM_CLASSES = 5
model = torchvision.models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, NUM_CLASSES)
model.fc.out_features


# Craete optimizer and loss
learning_rate = config["training"]["lr"]
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)


dataloaders = {'train': train_loader, 'val': val_loader}
model = train(model, dataloaders, criterion, optimizer, config)




