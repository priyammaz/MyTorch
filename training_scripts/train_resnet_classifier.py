import os
import argparse
import cupy as cp
import numpy as np
from tqdm import tqdm
import mytorch
import mytorch.nn as nn
import mytorch.optim as optim
from mytorch.utils.data import DataLoader
from models.resnet import ResNet50

### USE Torch for Torchvision Stuff! ###
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms

model = ResNet50()
print(model)

PATH_TO_DATA = "../../data/dogsvscats/"

### DEFINE TRANSFORMATIONS ###
normalizer = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) ### IMAGENET MEAN/STD ###
train_transforms = transforms.Compose([
                                        transforms.Resize((224,224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalizer
                                      ])


dataset = ImageFolder(PATH_TO_DATA, transform=train_transforms)
train_samples, test_samples = int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths=[train_samples, test_samples])

def collate_fn(batch):

    ### Prep and Scale Images ###
    images = cp.stack([cp.array(i[0]) for i in batch]) 

    ### One Hot Encode Label ###
    labels = [i[1] for i in batch]

    images = mytorch.Tensor(images).astype(cp.float32)
    labels = mytorch.Tensor(labels)

    return images, labels

trainloader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn, num_workers=2, shuffle=True)
testloader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn, num_workers=2, shuffle=True)

### Prep Optimizer ###
optimizer = optim.Adam(model.parameters(), lr=0.0001)

### Prep Loss Function ###
loss_fn = nn.CrossEntropyLoss()

### Train Model for 10 Epochs ###
for epoch in range(10):

    print(f"Training Epoch {epoch}")

    train_loss, train_acc = [], []
    eval_loss, eval_acc = [], []

    model.train()
    for images, labels in tqdm(trainloader):

        ### Pass Through Model ###
        pred = model(images)    

        ### Compute Loss ###
        loss = loss_fn(pred, labels)

        ### Compute Accuracy ###
        predicted = pred.argmax(dim=-1)
        accuracy = (predicted == labels).sum() / len(predicted)
        
        ### Log Results ###
        train_loss.append(loss.item())
        train_acc.append(accuracy.item())
        
        ### Update Model ###
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    model.eval()
    for images, labels in tqdm(testloader):

        with mytorch.no_grad():
            ### Pass Through Model ###
            pred = model(images)
            
        ### Compute Loss ###
        loss = loss_fn(pred, labels)

        ### Compute Accuracy ###
        predicted = pred.argmax(dim=-1)
        accuracy = (predicted == labels).sum() / len(predicted)

        eval_loss.append(loss.item())
        eval_acc.append(accuracy.item())

    print(f"Training Loss: {np.mean(train_loss)}")
    print(f"Eval Loss: {np.mean(eval_loss)}")
    print(f"Training Acc: {np.mean(train_acc)}")
    print(f"Eval Acc: {np.mean(eval_acc)}")