---
title: "[DL 101] Autoencoder Tutorial (Pytorch)"
excerpt: "Today we are going to build a simple autoencoder model using pytorch. We'll flatten CIFAR-10 dataset vectors then train the autoencoder with these flattened data."
date: 2021-02-20 15:000 -0400
author : 오승미
categories :
  - deep-learning
  - pytorch
tags :
  - deep-learning
  - autoencoder

---

# Autoencoder Tutorial

Today we are going to build a simple autoencoder model using pytorch. We'll flatten CIFAR-10 dataset into 3072(=32*3\*3) vectors then train the autoencoder with these flattened data.

## Load Libraries

```python
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

import matplotlib.pyplot as plt
import numpy as np
import easydict

import warnings
warnings.filterwarnings(action='ignore')

args = easydict.EasyDict({
    "SAVE_PATH":"./model",
    "NUM_CLASSES":10,
    "BATCH_SIZE":256,
    "NUM_EPOCHS":5,
    "FEATURE_EXTRACT":True,
    "LEARNING_RATE":0.0001,
    "WEIGHT_DECAY":0.005,
    "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
})
```



## Transform

```python
transform = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
            ])

train_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root='./data', 																									 train=True, download=True, transform=transform),
                                           batch_size=args['BATCH_SIZE'],
                                           shuffle=True,
                                           num_workers=4,
                                           drop_last=True)
valid_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root='./data', 																									 train=False, download=True, transform=transform),
                                           batch_size=args['BATCH_SIZE'],
                                           shuffle=False,
                                           num_workers=4,
                                           drop_last=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```



## Model

The thing is that we need to add nonlinear function such as ReLU enough, otherwise it just works as PCA. The encoder extracts features from dataset, which can be used in dimension reduction. On the other hand, the decoder reconstruct the original data.

```python
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Linear(3072, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.encoder3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.encoder3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.decoder3 = nn.Sequential(
            nn.Linear(256, 3072),
        )



    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        dec1 = self.decoder1(enc3)
        dec2 = self.decoder2(dec1)
        dec3 = self.decoder3(dec2)
        return enc3, dec3
```



## Train and Validation

```python
def validation(model, valid_loader, criterion):

    accuracy = 0
    valid_loss = 0

    for i, (inputs, _) in enumerate(valid_loader, 0):
        inputs = inputs.view(args['BATCH_SIZE'], -1)

        encoded, outputs = model(inputs)
        loss = criterion(outputs, inputs)
        valid_loss += loss.item()

    return valid_loss
```

```python
def train_model(model, train_loader, valid_loader, criterion, optimizer, args):

    steps = 0
    total_step = len(train_loader)
    train_losses, val_losses = [],[]

    for epoch in range(args['NUM_EPOCHS']):  

        running_loss = 0
        for i, (inputs, _) in enumerate(train_loader, 0):

            inputs = inputs.view(args['BATCH_SIZE'], -1)

            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = criterion(outputs, inputs)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % total_step == (total_step - 1):
                model.eval()
                with torch.no_grad():
                    valid_loss = validation(model, valid_loader, criterion)

                # print train loss, validation loss, validation accuracy
                print("epoch: {}/{}".format(epoch+1, args['NUM_EPOCHS']),
                      "train loss: {:.6f}".format(running_loss/total_step),
                      "val loss: {:.6f}".format(valid_loss/len(valid_loader)))

                # save model
                torch.save(model.state_dict(), 	args['SAVE_PATH']+"/checkpoint_"+str(epoch+1)+".tar")

                train_losses.append(running_loss/len(train_loader))
                val_losses.append(valid_loss/len(valid_loader))
                running_loss = 0
                model.train()

    return('Finished Training')

```

```python
model = AutoEncoder()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),
                       lr=args['LEARNING_RATE'],
                       weight_decay=args['WEIGHT_DECAY'])
```

```python
train_model(model, train_loader, valid_loader, criterion, optimizer, args)
```



## Reference

- https://blog.keras.io/building-autoencoders-in-keras.html
