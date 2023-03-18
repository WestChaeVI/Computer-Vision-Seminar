import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
from tensorflow import summary
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from torch import optim
from model import ResNet50
from cv_dataset import Custom_dataset as C
from data_loader import cv_loader


device = 'cuda' if torch.cuda.is_available() else 'cpu' # device 배정
torch.manual_seed(42)
if device == 'cuda':
  torch.cuda.manual_seed_all(42)

# 하이퍼파라미터
batch_size = 8
lr = 0.0001
epochs = 100
optimizer_name = 'adam'
model_name = 'resnet50'
criterion = nn.CrossEntropyLoss().to(device)

root_path = '/content/drive/MyDrive/CV_seminar_project'
train_path = '/content/drive/MyDrive/CV_seminar_project/train'
valid_path = '/content/drive/MyDrive/CV_seminar_project/valid'

train_loader, valid_loader, test_laoder = cv_loader()
resnet_50 = ResNet50()

optimizer = torch.optim.Adam(resnet_50.parameters(), lr = lr, weight_decay = 1e-8) # 학습을 할수록 학습률 낮춰주는 역할(러닝 보폭 줄이기, 섬세하게 보기위해)

train_acc_lst, train_loss_lst, test_acc_lst, test_loss_lst= [], [], [], []

epochs = 100
model_name = 'resnet50'
state={}

def Train():
    for epoch in range(1, epochs+1):

        train_loss = 0.0
        total = 0
        correct = 0
        train_acc = 0
  
        resnet_50.train()
        for i, (train_img, train_label) in enumerate(train_loader):
        # gpu에 할당
        train_img = train_img.to(device)
        train_label = train_label.to(device)

        output = resnet_50(train_img) # 모델에 입력

        optimizer.zero_grad( set_to_none = True ) # 계산했던 가중치 초기화    
        loss = criterion(output, train_label)
        loss.backward() # 미분
        optimizer.step() # 학습

        # loss & acc
        train_loss += loss.item()
        _, predictions = torch.max(output.data ,dim = 1)
   
        total += train_label.size(0)
        correct += (predictions == train_label).sum().item()
        train_acc += 100 * (correct / total)

        train_loss = round(train_loss/(i+1), 3) # 소수점 반올림
        train_acc = round(train_acc/(i+1), 3)
        print(f'Trainset {epoch}/{epochs} Loss : {train_loss}, Accuracy : {train_acc}%')
        train_acc_lst.append(train_acc)
        train_loss_lst.append(train_loss)

        # -------------------------------------------------------------------------------------
        test_loss = 0.0
        corrects = 0
        totals = 0
        test_acc = 0

        resnet_50.eval()
        with torch.no_grad():

        for i, (valid_img, valid_label) in enumerate(valid_loader):
                # gpu에 할당
            valid_img = valid_img.to(device)
            valid_label = valid_label.to(device)

            outputs = resnet_50(valid_img) # 모델에 입력
            losses = criterion(outputs, valid_label)

            # loss & acc
            test_loss += losses.item()
            _, predictions = torch.max(outputs.data ,dim = 1 )
    
            totals += valid_label.size(0)
            corrects += (predictions == valid_label).sum().item()
            test_acc += 100 * (corrects / totals)

        test_loss = round(test_loss/(i+1), 3) # 소수점 반올림
        test_acc = round(test_acc/(i+1), 3)
        print(f'Validset {epoch}/{epochs} Loss : {test_loss}, Accuracy : {test_acc}% \n')
        test_loss_lst.append(test_loss)
        test_acc_lst.append(test_acc)
  
        # 모델 저장
        if np.max(test_acc_lst) <= test_acc:

        state['epoch'] = epoch
        state['net'] = resnet_50.state_dict()

        state['train_loss'] = train_loss
        state['test_loss'] = test_loss

        state['train_acc'] = train_acc
        state['test_acc'] = test_acc
    torch.save(state, '/content/drive/MyDrive/CV_seminar_project/resnet50_{}_{}.pth'.format(str(state['epoch']), str(state['test_acc'])))