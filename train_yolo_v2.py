import pandas as pd
import numpy as np
import cv2
import math 
import glob
import os
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

import load_dataset_v2
import yolo_loss_v2
import models
#import yolo_loss_best
import models_best

filepath = './hw2_train_val/train15000/'
modelpath = './model_zoo'
device1 ='cuda:0'
device2 ='cuda:1'
batch_size = 50

yolo_net = models.Yolov1_vgg16bn(pretrained=True)
yolo_net_best = models_best.Yolov1_vgg16bn(pretrained=True)
#transform=T.Compose(T.ToTensor())
transform = T.Compose([T.ToTensor(),
                       T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
                      ])

trainset = load_dataset_v2.Dota_tarin(filepath,
                                      transform=transform,
                                      advanced=True, 
                                      threshold = 0.3)
trainset_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=14)

def train(model, model_b, epoch, log_interval=100, lr=1e-2, save_num=300):
    #baseline
    model.to(device1)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = yolo_loss_v2.YOLOloss(device1).to(device1)
    #best
    model_b.to(device2)
    model_b.train()
    optimizer_b = optim.Adam(model_b.parameters(), lr=lr)
    criterion_b = yolo_loss_v2.YOLOloss_best(device2).to(device2)
    
    try:
        load_checkpoint(os.path.join(modelpath,'yolo_baseline.pth'),model,optimizer)
    except:
        print('***yolo_baseline === no trained model ===')
    try:
        load_checkpoint(os.path.join(modelpath,'yolo_best.pth'),model_b,optimizer_b)
    except:
        print('***yolo_best === no trained model ===')
    
    iteration = 0
    for ep in range(epoch):
        t0 = time.time()
        lbase,lbest = 0.,0.
        for batch_idx, (data, target,target_b, fn) in enumerate(trainset_loader):
            #baseline
            data_a, target = data.clone().to(device1), target.to(device1)  
            optimizer.zero_grad()
            output = model(data_a)
            loss = criterion(output.float(), target.float())
            loss.backward()
            optimizer.step()
            #best
            data_b, target_b = data.clone().to(device2), target_b.to(device2)  
            optimizer_b.zero_grad()
            output_b = model_b(data_b)
            loss_b = criterion_b(output_b.float(), target_b.float())
            loss_b.backward()
            optimizer_b.step()
            
            lbase += loss.item()
            lbest += loss_b.item()
            
            if iteration % log_interval == 0:
                print('Epoch: {} [{}/{} ({:.0f}%)] BaseLoss:{:.4f} BestLoss:{:.4f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item(),loss_b.item()))                
            iteration += 1
        print('======================')
        print('{:.1f} secs ,Base {:.4f}, Best {:.4f}'.format(time.time() - t0,
              batch_size*lbase/15000,batch_size*lbest/15000))
        print('======================')
        t0 = time.time()
        if ep % 2 == 0:
            save_checkpoint('yolo_baseline.pth', model, optimizer)
            save_checkpoint('yolo_best.pth', model_b, optimizer_b)
#        if ep % 50 == 0:
#            name = 'yolo_baseline_epo_'+str(ep+save_num)+'.pth'
#            save_checkpoint(name, model, optimizer)
#            name = 'yolo_best_epo_'+str(ep+save_num)+'.pth'
#            save_checkpoint(name, model_b, optimizer_b)

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, os.path.join(modelpath,checkpoint_path))
    name_='BW_'+checkpoint_path
    torch.save(model,os.path.join(modelpath,name_))
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)
            
train(yolo_net,yolo_net_best ,epoch=1,log_interval=int(5000/batch_size), lr=1e-4, save_num=0)
torch.save(yolo_net,os.path.join(modelpath,'BW_yolo_baseline_1.pth'))
torch.save(yolo_net_best,os.path.join(modelpath,'BW_yolo_best_1.pth'))
train(yolo_net,yolo_net_best ,epoch=10,log_interval=int(5000/batch_size), lr=1e-4, save_num=0)
torch.save(yolo_net,os.path.join(modelpath,'BW_yolo_baseline_11.pth'))
torch.save(yolo_net_best,os.path.join(modelpath,'BW_yolo_best_11.pth'))
