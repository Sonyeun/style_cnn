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

# 데이터셋 다운받을 경로를 지정하고, 데이터셋을 불러옵니다.
# specify the data path
path2data = 'C:\\Users\\User\\Desktop\\project\\style_cnn\\data'


# if not exists the path, make the directory
if not os.path.exists(path2data):
    os.mkdir(path2data)

# load dataset
#STL-10 dataset은 96x96 크기의 RGB 컬러 이미지이며, 5000개 train 이미지, 8,000개 test 이미지
train_ds = datasets.STL10(path2data, split='train', download=True, transform=transforms.ToTensor())
val_ds = datasets.STL10(path2data, split='test', download=True, transform=transforms.ToTensor())
print(len(train_ds))
print(len(val_ds))

# To normalize the dataset, calculate the mean and std
# It converts a tensor object into an numpy.ndarray object.// 각 채널값들의 평균
train_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in train_ds] #RGB값, 라벨 -> [array(이미지1의 R평균, G평균, B평균),...]
train_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in train_ds]  #axis=m: m축을 없애겠다
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

# apply transforamtion다운받은 dataset은 이렇게 관리하나보다
train_ds.transform = train_transformation
val_ds.transform = val_transformation

# create DataLoader(TensorDataset을 DataLoader에 전달하면 for 루프에서 데이터의 일부분만 간단히 추출할 수 있게 된다)
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

#보여질 샘플 번호를 임의로 설정
grid_size = 4
rnd_inds = np.random.randint(0, len(train_ds), grid_size)
print('image indices:',rnd_inds)

#make_grid는 출력 사진들을 한번에 보여줄 좋은 방법
x_grid = [train_ds[i][0] for i in rnd_inds]
y_grid = [train_ds[i][1] for i in rnd_inds]
x_grid = utils.make_grid(x_grid, nrow = grid_size, padding = 10)




#모델 구축하기 ~ ResNet은 residual block이 겹겹이 쌓여 구성된 모델
#residual block 정의
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # BatchNorm에 bias가 포함되어 있으므로, 
        # conv2d는 bias=False로 설정합니다.

        #(필터 채널 수, 필터 개수, 필터 크기)
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size = 3,
                      stride = stride, padding = 1, bias = False),

            #각 배치 단위 별로 데이터가 다양한 분포를 가지더라도 
            #각 배치별로 평균과 분산을 이용해 정규화
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion,
                      kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

        # identity mapping
        # input과 outpu의 feature map size, filter 수가 동일한 경우
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        #projection mapping using 1*1conv
        #f(x) + x 하려는데 크기가 또는 채널 개수가 안 맞는 경우 x를 f(x)에 맞춤
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



#bottleneck 만들기
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

#### ResNet 구현#######
class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes = 10,
                 init_weights = True):
        """
        block은 사용할 block의 종류를 알려줌 ~ basic or bottleneck
        num_block은 list로 각conv층에서 몇개의 block이 있어야하는 지 알려줌
        """

        #pytorch에서 class 형태의 모델은 
        #항상 nn.Module을 상속 받아야 하며, 이를 실행시키기
        #기반 클래스의 속성을 subclass가 받아오도록
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

        #이거 이지랄 해도 되는 거야?
        #512*block.expansion개차원에서 각 하나인데?
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

        if init_weights:
            self._initialize_weights()


    #층을 구성하는 함수
    def _make_layer(self, block, out_channels, num_blocks, stride):
        #stride, 1, 1, .. 해당 층에서 몇개의 conv <-이게 왜 필요하지?
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

        #데이터 개수대로 일렬로
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        #self.modules() -> 모델 클래스에서 정의된 layer들을 
        #iterable로 차례로 반환
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #시작으로서 괜찮은 가중치 초기화 방식이다만 알고 가셈
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out',
                                        nonlinearity='relu')
                
                if m.bias is not None:
                    #입력으로 들어온 tensor의 값을 모두 0으로
                    nn.init.constant_(m.bias, 0)


            elif isinstance(m, nn.BatchNorm2d):
                #정규화하고 w,b 다 있음
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


##### 모델 학습하기 #######

#loss_func, optimizer 정의
#mean은 loss 평균, sum은 loss 합
loss_func = nn.CrossEntropyLoss(reduction = 'sum')
opt = optim.Adam(model.parameters(), lr = 0.001)

from torch.optim.lr_scheduler import ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(opt, mode = 'min', factor = 0.1, patience=10)


 #현재 lr을 계산하는 함수 정의
def get_lr(opt):
    #왜 굳이 이렇게 하는지는 몰겄다
    #param_groups[0]['lr']하면 되지 않나?
    for param_group in opt.param_groups:
        return param_group['lr']



###################################
#####################################
########여기서부터#################


# function to calculate metric per mini-batch
#(학습에 직접적으로 사용되지는 않지만 학습이 잘 되었는지 객관적으로 평가할 수 있는 지표)
def metric_batch(output, target):
    #output은 2차원(batch 개수, 각 확률)
    pred = output.argmax(1, keepdim = True) #차원 수를 유지(기존대로라면 한차원씩 줄어듦)
    
    #pred 배열과 target배열 중 일치하는 것 있는지 검사
    # 뒤에 .sum()을 붙임으로써 일치하는 것들의 개수의 합 출력
    #.item() 텐서에서 값만 가져옴
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
    
    ########이거 왜 이케 하냐?###########
    #그리고 yb 원핫인코딩 해줘야 되는 거 아니야?
    #데이터 어덯게 생겼는지 한번 확인 ㄱㄱ
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


# 학습을 시작하는 함수를 정의합니다.
#주석 처리한부분은 val_loss가 가장 낮을 때 모델의 가중치 저장
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

    #학습 시작
    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print(f'Epoch {epoch}/{num_epochs-1}, current_lr = {current_lr}')

        #여기에서 loss, backward, optimize 다 함
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        model.eval()
        # autograd engine을 꺼 ->연산량 줄임
        #with문 ~ 선언 expression을 생성 사용 반납(자동)
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
        #그냥 metric이 step에 들어가야 돼
        #ReduceLROnPlateau은 변화가 없을 때 움직이니까
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



