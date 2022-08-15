# -*- coding: cp949 -*-
# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import optim
from torch.optim.lr_scheduler import StepLR

import time
import copy
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# DenseNet BottleNeck
#residual과 shortcut을 add가 아닌 concat으로 연결
#채널 수를 확장하기에 효율적인 방법
class BottleNeck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()

        ####이거 머야?####
        inner_channels = 4*growth_rate
        
        self.residual = nn.Sequential(
            #pre-activate가 좋더라
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),

            #input_feature_maps를 줄임 1*1 convolution으로 ->computational efficiency
            #conv2d는 (batch, channel, h,w) 이렇게 받음
            nn.Conv2d(in_channels, inner_channels, kernel_size = 1,
                      stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
            nn.Conv2d(inner_channels, growth_rate, 3,
                      stride = 1, padding = 1, bias = False)
            )

        self.shortcut = nn.Sequential()

    def forward(self, x):
        #원하는 dimension 방향으로 텐서를 나란하게 쌓아줌
        #(batch, channel, h , w)형태로 들어가네
        return torch.cat([self.shortcut(x), self.residual(x)], dim = 1)


 #Transition Block
class Transition(nn.Module):
    """
    reduce feature map size and number of channels"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1,
            stride = 1, padding = 0, bias = False),
            nn.AvgPool2d(2, stride = 2)
            )

    def forward(self, x):
        return self.down_sample(x)
        

#전체 구조 정의하기
#중간중간 innel channel이 계속해서 바뀌는 것 조심
#(이전 레이어의 피쳐맵이 다음 모든 레이어에 합쳐지기에)
class DenseNet(nn.Module):
    #화가 총 47
    def __init__(self, nblocks, growth_rate = 12,
                reduction = 0.5, num_classes = 47, init_weights = True):
        super().__init__()


        self.growth_rate = growth_rate
        # output channels of conv1 before entering Dense Block
        inner_channels = 2 * growth_rate

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, inner_channels, 7, stride=2, padding=3),
            nn.MaxPool2d(3, 2, padding=1)
        )
        self.features = nn.Sequential()

        #[6,12,24,6]
        #i = 0인경우
        #그 수만큼 denseblock을 넣고
        #결과로 나오는 inner_channles를 줄이도록 transition 넣음
        for i in range(len(nblocks) - 1):
            self.features.add_module(f'dense_block_{i}',
                                     self._make_dense_block(nblocks[i], inner_channels))
            
            inner_channels += growth_rate * nblocks[i]
            out_channels = int(reduction * inner_channels)

            self.features.add_module('transition_layer_{}'.format(i), Transition(inner_channels, out_channels))
            inner_channels = out_channels 
        

        self.features.add_module(f'dense_block_{len(nblocks)-1}',
                                 self._make_dense_block(nblocks[len(nblocks) -1],
                                                                inner_channels))
        
        inner_channels += growth_rate * nblocks[len(nblocks) -1]

        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU())

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(inner_channels, num_classes)


        #weight initialization
        if init_weights:
            self._initialize_weights()


    def forward(self, x):
        #들어오는 x는 어떤 형태이지?
        try:
            #print('처음 x에 들어오는 형태')
            #print(x.size())
            #print('')
            #print('-----------------')
            x = self.conv1(x)
            x = self.features(x)
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.linear(x)

            return x 

        except:
            pass

    def _make_dense_block(self, nblock, inner_channels):
        dense_block = nn.Sequential()
        for i in range(nblock):
            dense_block.add_module('bottle_neck_layer_{}'.format(i), BottleNeck(inner_channels, self.growth_rate))
            
            #denseblock 하나 지나면 growth_rate만큼 증가
            inner_channels += self.growth_rate
        return dense_block

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def DenseNet_121():
    return DenseNet([6, 12, 24, 6])



################ 학습 ##############
#get current lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


# calculate the metric per mini-batch
def metric_batch(output, target):
    pred = output.argmax(1, keepdim = True)
    corrects = pred.eq(target.view_as(pred)).sum().item()

    return corrects

# calculate the loss per mini-batch
def loss_batch(loss_func, output, target, opt = None):
    """
    opt이 주어진다면 학습도 겸합니다"""
    try:
        loss_b = loss_func(output, target)
        metric_b = metric_batch(output, target)

    except:
        return 0, 0

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()

    return loss_b, metric_b

# calculate the loss per epochs
def loss_epoch(model, loss_func, dataset_dl, sanity_check = False, opt= None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl)

    #
    print('현재 들어온 batch of epoch 길이:', len(dataset_dl))
    #print('이거 되나? dataset_dl.size()', dataset_dl.size()) 안돼
    for xb, yb in dataset_dl:
        #print('이거 되나? xb.size()', xb.size()) 
        # ->torch.Size([32, 3, 64, 64])

        #print('이거 되나? yb.size()', yb.size())
        # -> torch.Size([32])

        xb = xb.to(device)
        yb = yb.to(device)

        try:
            output = model(xb)
            loss_b, metric_b = loss_batch(loss_func, output, yb,opt)
            running_loss += loss_b

            if metric_b is not None:
                running_metric += metric_b

            if sanity_check is True:
                break

        except:
            pass

    loss = running_loss / len_data
    metric = running_metric / len_data
    return loss, metric

#function to start training
def train_val(model, params):
    num_epochs=params['num_epochs']
    loss_func=params['loss_func']
    opt=params['optimizer']
    train_path=params['train_path']
    val_path=params['val_path']
    sanity_check=params['sanity_check']
    lr_scheduler=params['lr_scheduler']
    path2weights=params['path2weights']

    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    #최초 학습 시행 시에는 이미 있던 weight들을 가져오도록
    #왜냐하면 난 중간중간에 계속 학습을 끊어야되는 사정이 있어
    try:
        model.load_state_dict(torch.load(path2weights))
    except:
        pass


    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    start_time = time.time()

    print('학습 시작하겠습니다')
    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr= {}'.format(epoch, num_epochs-1, current_lr))

        train_loss, train_metric = 0.0, 0.0
        val_loss, val_metric = 0.0, 0.0 

        #수정이 필요햠
        model.train()

        with open(train_path, 'rb') as fs:

            ##one_epoch loss, metric 구하기##
            while True:
                try:
                    train_dl = pickle.load(fs)
                    via_loss, via_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
                    train_loss += via_loss
                    train_metric += via_metric
                    

                except:
                    break
            ##one_epoch loss, metric 구하기 끝 ##

            loss_history['train'].append(int(train_loss))
            metric_history['train'].append(int(train_metric))


        model.eval()
        with open(val_path, 'rb') as fs:
            
            ##one_epoch loss, metric 구하기##
            while True:
                try:
                    val_dl = pickle.load(fs)
                    with torch.no_grad():
                        via_loss, via_metric = loss_epoch(model, loss_func, val_dl, sanity_check)
                        val_loss += via_loss
                        val_metric += via_metric


                except:
                    break

            loss_history['val'].append(int(val_loss))
            metric_history['val'].append(int(val_metric))
            ##one_epoch loss, metric 구하기 끝##

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), path2weights)
                print('Copied best model weights!')


            #lr이 갱신될 경우, best model에서 다시 학습 시도
            lr_scheduler.step(val_loss)
            if current_lr != get_lr(opt):
                print('Loading best model weights!')
                model.load_state_dict(best_model_wts)

        print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' %(train_loss, val_loss, 100*val_metric, (time.time()-start_time)/60))
        print('-'*10)

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history