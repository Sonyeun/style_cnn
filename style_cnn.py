# -*- coding: cp949 -*-
# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import os

##util
import numpy as np
from torchsummary import summary
import time
import copy
import glob
from PIL import Image
import random
import pickle

#densenet 불러오기
import densenet

# display images
from torchvision import utils
import matplotlib.pyplot as plt



# 모델 만들기 ~ 현재는 (1,1)의 사이즈로 최종 pooling 됨
#make model and check model if u want
model = densenet.DenseNet_121()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#학습하기
#define loss function, optimizer, lr_scheduler
loss_func = nn.CrossEntropyLoss(reduction = 'sum')
opt = optim.Adam(model.parameters(), lr = 0.01)
lr_scheduler = ReduceLROnPlateau(opt, mode = 'min', factor = 0.1, patience = 8)

train_path = 'C:\\Users\\User\\Desktop\\origin\\train_dl.pkl'
val_path = 'C:\\Users\\User\\Desktop\\origin\\val_dl.pkl'

# define the training parameters
params_train = {
    'num_epochs':10,
    'optimizer':opt,
    'loss_func':loss_func,
    'train_path':train_path,
    'val_path':val_path,
    'sanity_check':False,
    'lr_scheduler':lr_scheduler,
    'path2weights':'./models/weights.pt'
,
}

model, loss_hist, metric_hist = densenet.train_val(model, params_train)


# Train-Validation progress
num_epochs = params_train['num_epochs']

# Train-Validation progress
num_epochs = params_train['num_epochs']

# plot loss progress
plt.title('Train-Val Loss')
plt.plot(range(1, num_epochs + 1), loss_hist['train'], label='train')
plt.plot(range(1, num_epochs + 1), loss_hist['val'], label='val')
plt.ylabel('Loss')
plt.xlabel('Training Epochs')
plt.legend()
plt.show()

# plot accuracy progress
plt.title('Train-Val Accuracy')
plt.plot(range(1, num_epochs + 1), metric_hist['train'], label='train')
plt.plot(range(1, num_epochs + 1), metric_hist['val'], label='val')
plt.ylabel('Accuracy')
plt.xlabel('Training Epochs')
plt.legend()
plt.show()




















