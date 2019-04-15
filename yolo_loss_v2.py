import pandas as pd
import numpy as np
import cv2
import math 
import glob
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import dataloader, dataset
import torch.utils.model_zoo as model_zoo
import torchvision
import torchvision.transforms as T

#device ='cuda:0'

class YOLOloss(nn.Module):
    def __init__(self,device):
        super(YOLOloss,self).__init__()
        self.device = device
        
    def IOU(self,box_pred,box_targ):
        #input size = [box_n,4]  target-1-to-1-predict
        #box_target num = box_n 
        box_n=box_targ.shape[0]
        #recover to normal size ratio
        area_pred = 7*7*torch.prod(box_pred[:,[2,3]],1) #w*h
        area_targ = 7*7*torch.prod(box_targ[:,[2,3]],1) #w*h
        Diff = torch.abs(box_pred-box_targ)
        Diff[:,[2,3]] *=3.5
        Summ = 3.5*(box_pred+box_targ)
        
        Status_x = Diff[:,0]>Diff[:,2]
        Status_y = Diff[:,1]>Diff[:,3]
        
        Ix = torch.zeros(box_n).float().to(self.device)
        Iy = torch.zeros(box_n).float().to(self.device)
        
        if (~Status_x).sum()>0:
            Ix[~Status_x]=7*(torch.min(box_pred[:,2],box_targ[:,2]))[~Status_x]
        if (~Status_y).sum()>0:
            Iy[~Status_y]=7*(torch.min(box_pred[:,3],box_targ[:,3]))[~Status_y]
        if (Status_x).sum()>0:
            Ix[Status_x]=(torch.clamp(Summ[:,2]-Diff[:,0],0))[Status_x]
        if (Status_y).sum()>0:
            Iy[Status_y]=(torch.clamp(Summ[:,3]-Diff[:,1],0))[Status_y]

        area_Inter = Ix*Iy
        
        out_IOU = area_Inter/(area_pred+area_targ-area_Inter+1e-6)
#        out_IOU = out_IOU.view(-1,1)
        return out_IOU
    
    def forward(self,predict,target):
        #inpit :,7,7,26 -> flat :*7*7,26
        pred_tensor = predict.view(-1,26)
        targ_tensor = target.view(-1,26)       
        #mask obj
        obj1 = targ_tensor[:,4]>0
        obj0 = ~obj1
        #filter by mask obj
        obj1_pred = pred_tensor[obj1]
        obj1_targ = targ_tensor[obj1]
        obj0_pred = pred_tensor[obj0]
        obj0_targ = targ_tensor[obj0]
        #split to Box and class        
        obj1_pred_class = obj1_pred[:,10:]
        obj1_targ_class = obj1_targ[:,10:]
        obj1_pred_boxes = obj1_pred[:,:10]
        obj1_targ_boxes = obj1_targ[:,:10]
        obj0_pred_boxes = obj0_pred[:,:10]
        obj0_targ_boxes = obj0_targ[:,:10]
        #falt boxes to all box flat to :*7*7*2,5
        obj1_pred_box = obj1_pred_boxes.contiguous().view(-1,5)
        obj1_targ_box = obj1_targ_boxes.contiguous().view(-1,5)
        obj0_pred_box = obj0_pred_boxes.contiguous().view(-1,5)
        obj0_targ_box = obj0_targ_boxes.contiguous().view(-1,5)
        #cal IOU by function
        iou_box = self.IOU(obj1_pred_box,obj1_targ_box)
        #TARGET COnfidence = 1*IOU
#        obj1_targ_box[:,4] *= iou_box
        #Cal loss
        loss_xy = F.mse_loss(obj1_pred_box[:,[0,1]],obj1_targ_box[:,[0,1]])
        loss_wh = F.mse_loss((7*obj1_pred_box[:,[2,3]])**0.5,(7*obj1_targ_box[:,[2,3]])**0.5)
        loss_obj1_c = F.mse_loss(obj1_pred_box[:,4],iou_box)
        loss_obj0_c = F.mse_loss(obj0_pred_box[:,4],obj0_targ_box[:,4])
        loss_class = F.mse_loss(obj1_pred_class,obj1_targ_class)
        
        total_loss = 5*(loss_xy+loss_wh)+1*loss_obj1_c+0.5*loss_obj0_c+loss_class
        
        return total_loss

#device ='cuda:1'

class YOLOloss_best(nn.Module):
    def __init__(self,device):
        super(YOLOloss_best,self).__init__()
        self.device = device
        
    def IOU(self,box_pred,box_targ):
        #input size = [box_n,4]  target-1-to-1-predict
        #box_target num = box_n 
        box_n=box_targ.shape[0]
        #recover to normal size ratio
        area_pred = 7*7*torch.prod(box_pred[:,[2,3]],1) #w*h
        area_targ = 7*7*torch.prod(box_targ[:,[2,3]],1) #w*h
        Diff = torch.abs(box_pred-box_targ)
        Diff[:,[2,3]] *=3.5
        Summ = 3.5*(box_pred+box_targ)
        
        Status_x = Diff[:,0]>Diff[:,2]
        Status_y = Diff[:,1]>Diff[:,3]
        
        Ix = torch.zeros(box_n).float().to(self.device)
        Iy = torch.zeros(box_n).float().to(self.device)
        
        if (~Status_x).sum()>0:
            Ix[~Status_x]=7*(torch.min(box_pred[:,2],box_targ[:,2]))[~Status_x]
        if (~Status_y).sum()>0:
            Iy[~Status_y]=7*(torch.min(box_pred[:,3],box_targ[:,3]))[~Status_y]
        if (Status_x).sum()>0:
            Ix[Status_x]=(torch.clamp(Summ[:,2]-Diff[:,0],0))[Status_x]
        if (Status_y).sum()>0:
            Iy[Status_y]=(torch.clamp(Summ[:,3]-Diff[:,1],0))[Status_y]

        area_Inter = Ix*Iy
        
        out_IOU = area_Inter/(area_pred+area_targ-area_Inter+1e-6)
#        out_IOU = out_IOU.view(-1,1)
        return out_IOU
    
    def forward(self,predict,target):
        #inpit :,7,7,63 -> flat :*7*7,21
        pred_tensor = predict.view(-1,21)
        targ_tensor = target.view(-1,21)       
        #mask obj
        obj1 = targ_tensor[:,4]>0
        obj0 = ~obj1
        #filter by mask obj
        obj1_pred = pred_tensor[obj1]
        obj1_targ = targ_tensor[obj1]
        obj0_pred = pred_tensor[obj0]
        obj0_targ = targ_tensor[obj0]
        #split to Box and class        
        obj1_pred_class = obj1_pred[:,5:]
        obj1_targ_class = obj1_targ[:,5:]
        obj1_targ_cls = torch.argmax(obj1_targ_class,1)
        obj1_pred_box = obj1_pred[:,:5]
        obj1_targ_box = obj1_targ[:,:5]
        obj0_pred_box = obj0_pred[:,:5]
        obj0_targ_box = obj0_targ[:,:5]
        #cal IOU by function
        iou_box = self.IOU(obj1_pred_box,obj1_targ_box)
        #TARGET COnfidence = 1*IOU
        #Cal loss
        loss_xy = F.mse_loss(obj1_pred_box[:,[0,1]],obj1_targ_box[:,[0,1]])
        loss_wh = F.mse_loss((7*obj1_pred_box[:,[2,3]])**0.5,(7*obj1_targ_box[:,[2,3]])**0.5)
        loss_obj1_c = F.mse_loss(obj1_pred_box[:,4],iou_box)
        loss_obj0_c = F.mse_loss(obj0_pred_box[:,4],obj0_targ_box[:,4])
        loss_class = F.mse_loss(obj1_pred_class,obj1_targ_class)
#        loss_class = F.cross_entropy(obj1_pred_class,obj1_targ_cls)        
        total_loss = 5*(loss_xy+loss_wh)+1*loss_obj1_c+0.5*loss_obj0_c+loss_class
        
        return total_loss