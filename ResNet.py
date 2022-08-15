# -*- coding: cp949 -*-
# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import optim
from torch.optim.lr_scheduler import StepLR

# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# display images
from torchvision import utils
import matplotlib.pyplot as plt

import numpy as np
from torchsummary import summary
import time
import copy

# �����ͼ� �ٿ���� ��θ� �����ϰ�, �����ͼ��� �ҷ��ɴϴ�.
# specify the data path
path2data = 'C:\\Users\\User\\Desktop\\project\\style_cnn\\data'


# if not exists the path, make the directory
if not os.path.exists(path2data):
    os.mkdir(path2data)

# load dataset
#STL-10 dataset�� 96x96 ũ���� RGB �÷� �̹����̸�, 5000�� train �̹���, 8,000�� test �̹���
train_ds = datasets.STL10(path2data, split='train', download=True, transform=transforms.ToTensor())
val_ds = datasets.STL10(path2data, split='test', download=True, transform=transforms.ToTensor())
print(len(train_ds))
print(len(val_ds))

# To normalize the dataset, calculate the mean and std
# It converts a tensor object into an numpy.ndarray object.// �� ä�ΰ����� ���
train_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in train_ds] #RGB��, �� -> [array(�̹���1�� R���, G���, B���),...]
train_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in train_ds]  #axis=m: m���� ���ְڴ�
train_meanR = np.mean([m[0] for m in train_meanRGB])
train_meanG = np.mean([m[1] for m in train_meanRGB])
train_meanB = np.mean([m[2] for m in train_meanRGB])
train_stdR = np.mean([s[0] for s in train_stdRGB])
train_stdG = np.mean([s[1] for s in train_stdRGB])
train_stdB = np.mean([s[2] for s in train_stdRGB])

##
val_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in val_ds]
val_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in val_ds]
val_meanR = np.mean([m[0] for m in val_meanRGB])
val_meanG = np.mean([m[1] for m in val_meanRGB])
val_meanB = np.mean([m[2] for m in val_meanRGB])
val_stdR = np.mean([s[0] for s in val_stdRGB])
val_stdG = np.mean([s[1] for s in val_stdRGB])
val_stdB = np.mean([s[2] for s in val_stdRGB])


# define the image transformation
train_transformation = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize(224),
                        transforms.Normalize([train_meanR, train_meanG, train_meanB],[train_stdR, train_stdG, train_stdB]),
                        transforms.RandomHorizontalFlip(),
])

val_transformation = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize(224),
                        transforms.Normalize([train_meanR, train_meanG, train_meanB],[train_stdR, train_stdG, train_stdB]),
])

# apply transforamtion�ٿ���� dataset�� �̷��� �����ϳ�����
train_ds.transform = train_transformation
val_ds.transform = val_transformation

# create DataLoader(TensorDataset�� DataLoader�� �����ϸ� for �������� �������� �Ϻκи� ������ ������ �� �ְ� �ȴ�)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=True)


# display sample images 
def show(img, y= None, color = True):
    npimg = img.numpy()
    npimg_tr = np.transpose(npimg, (1,2,0))
    plt.imshow(npimg_tr)

    if y is not None:
        plt.title('labels: ' + str(y))

np.random.seed(1)
torch.manual_seed(1)

#������ ���� ��ȣ�� ���Ƿ� ����
grid_size = 4
rnd_inds = np.random.randint(0, len(train_ds), grid_size)
print('image indices:',rnd_inds)

#make_grid�� ��� �������� �ѹ��� ������ ���� ���
x_grid = [train_ds[i][0] for i in rnd_inds]
y_grid = [train_ds[i][1] for i in rnd_inds]
x_grid = utils.make_grid(x_grid, nrow = grid_size, padding = 10)




#�� �����ϱ� ~ ResNet�� residual block�� ����� �׿� ������ ��
#residual block ����
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # BatchNorm�� bias�� ���ԵǾ� �����Ƿ�, 
        # conv2d�� bias=False�� �����մϴ�.

        #(���� ä�� ��, ���� ����, ���� ũ��)
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size = 3,
                      stride = stride, padding = 1, bias = False),

            #�� ��ġ ���� ���� �����Ͱ� �پ��� ������ �������� 
            #�� ��ġ���� ��հ� �л��� �̿��� ����ȭ
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion,
                      kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

        # identity mapping
        # input�� outpu�� feature map size, filter ���� ������ ���
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        #projection mapping using 1*1conv
        #f(x) + x �Ϸ��µ� ũ�Ⱑ �Ǵ� ä�� ������ �� �´� ��� x�� f(x)�� ����
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BasicBlock.expansion,
                kernel_size=1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels*BasicBlock.expansion)
                )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x



#bottleneck �����
class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                      stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels*BottleNeck.expansion,
                      kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion)
        )

        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        if stride !=1 or in_channels != out_channels*BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expansion,
                          kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
                )

    def forward(self,x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x

#### ResNet ����#######
class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes = 10,
                 init_weights = True):
        """
        block�� ����� block�� ������ �˷��� ~ basic or bottleneck
        num_block�� list�� ��conv������ ��� block�� �־���ϴ� �� �˷���
        """

        #pytorch���� class ������ ���� 
        #�׻� nn.Module�� ��� �޾ƾ� �ϸ�, �̸� �����Ű��
        #��� Ŭ������ �Ӽ��� subclass�� �޾ƿ�����
        super().__init__()

        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size = 7, stride = 2,
                      padding = 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0],1)
        self.conv3_x = self._make_layer(block, 128, num_block[1],2)
        self.conv4_x = self._make_layer(block, 256, num_block[2],2)
        self.conv5_x = self._make_layer(block, 512, num_block[3],2)

        #�̰� ������ �ص� �Ǵ� �ž�?
        #512*block.expansion���������� �� �ϳ��ε�?
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

        if init_weights:
            self._initialize_weights()


    #���� �����ϴ� �Լ�
    def _make_layer(self, block, out_channels, num_blocks, stride):
        #stride, 1, 1, .. �ش� ������ ��� conv <-�̰� �� �ʿ�����?
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels,stride))
            self.in_channels = out_channels*block.expansion

        return nn.Sequential(*layers)

    def forward(self,x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)

        #������ ������� �Ϸķ�
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        #self.modules() -> �� Ŭ�������� ���ǵ� layer���� 
        #iterable�� ���ʷ� ��ȯ
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #�������μ� ������ ����ġ �ʱ�ȭ ����̴ٸ� �˰� ����
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out',
                                        nonlinearity='relu')
                
                if m.bias is not None:
                    #�Է����� ���� tensor�� ���� ��� 0����
                    nn.init.constant_(m.bias, 0)


            elif isinstance(m, nn.BatchNorm2d):
                #����ȭ�ϰ� w,b �� ����
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    return ResNet(BottleNeck, [3,4,6,3])

def resnet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet50().to(device)
x = torch.randn(3, 3, 224, 224).to(device)
output = model(x)

print(output.size())
summary(model, (3, 224, 224), device=device.type)


##### �� �н��ϱ� #######

#loss_func, optimizer ����
#mean�� loss ���, sum�� loss ��
loss_func = nn.CrossEntropyLoss(reduction = 'sum')
opt = optim.Adam(model.parameters(), lr = 0.001)

from torch.optim.lr_scheduler import ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(opt, mode = 'min', factor = 0.1, patience=10)


 #���� lr�� ����ϴ� �Լ� ����
def get_lr(opt):
    #�� ���� �̷��� �ϴ����� ���δ�
    #param_groups[0]['lr']�ϸ� ���� �ʳ�?
    for param_group in opt.param_groups:
        return param_group['lr']



###################################
#####################################
########���⼭����#################


# function to calculate metric per mini-batch
#(�н��� ���������� �������� ������ �н��� �� �Ǿ����� ���������� ���� �� �ִ� ��ǥ)
def metric_batch(output, target):
    #output�� 2����(batch ����, �� Ȯ��)
    pred = output.argmax(1, keepdim = True) #���� ���� ����(������ζ�� �������� �پ��)
    
    #pred �迭�� target�迭 �� ��ġ�ϴ� �� �ִ��� �˻�
    # �ڿ� .sum()�� �������ν� ��ġ�ϴ� �͵��� ������ �� ���
    #.item() �ټ����� ���� ������
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects



# function to calculate loss per mini-batch
def loss_batch(loss_func, output, target, opt= None):
    loss = loss_func(output, target)
    metric_b = metric_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b

# function to calculate loss and metric per epoch
def loss_epoch(model, loss_func, dataset_dl, 
               sanity_check = False, opt = None):
    """
    function to calculate loss and metric per epoch
    """
    running_loss = 0.0 
    running_metric = 0.0
    
    ########�̰� �� ���� �ϳ�?###########
    #�׸��� yb �������ڵ� ����� �Ǵ� �� �ƴϾ�?
    #������ ��F�� ������� �ѹ� Ȯ�� ����
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)

        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
        running_loss += loss_b

        if metric_b is not None:
            running_metric += metric_b

        if sanity_check is True:
            break

    loss = running_loss/len_data
    metric = running_metric / len_data

    return loss, metric


# �н��� �����ϴ� �Լ��� �����մϴ�.
#�ּ� ó���Ѻκ��� val_loss�� ���� ���� �� ���� ����ġ ����
def train_val(model, params):
    num_epochs=params['num_epochs']
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    sanity_check=params["sanity_check"]
    lr_scheduler=params["lr_scheduler"]
    path2weights=params["path2weights"]

    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    # # GPU out of memoty error
    # best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = float('inf')

    start_time = time.time()

    #�н� ����
    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print(f'Epoch {epoch}/{num_epochs-1}, current_lr = {current_lr}')

        #���⿡�� loss, backward, optimize �� ��
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        model.eval()
        # autograd engine�� �� ->���귮 ����
        #with�� ~ ���� expression�� ���� ��� �ݳ�(�ڵ�)
        #loop ~ every tensor inside the loop will have requires_grad set to False
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss

            #best_model_wts = copy.deepcopy(model.state_dict())          
            #torch.save(model.state_dict(), path2weights)
            #print('Copied best model weights)
            print('Get best val_loss')

        ##########
        #�׳� metric�� step�� ���� ��
        #ReduceLROnPlateau�� ��ȭ�� ���� �� �����̴ϱ�
        lr_scheduler.step(train_loss)

        print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' %(train_loss, val_loss, 100*val_metric, (time.time()-start_time)/60))
        print('-'*10)

        return model, loss_history, metric_history


# define the training parameters
params_train = {
    'num_epochs':20,
    'optimizer':opt,
    'loss_func':loss_func,
    'train_dl':train_dl,
    'val_dl':val_dl,
    'sanity_check':False,
    'lr_scheduler':lr_scheduler,
    'path2weights':'./models/weights.pt',
}

# create the directory that stores weights.pt
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSerror:
        print('Error')


createFolder('./models')

model, loss_hist, metric_hist = train_val(model, params_train)

# Train-Validation Progress
num_epochs=params_train["num_epochs"]

# plot loss progress
plt.title("Train-Val Loss")
plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

# plot accuracy progress
plt.title("Train-Val Accuracy")
plt.plot(range(1,num_epochs+1),metric_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),metric_hist["val"],label="val")
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()


