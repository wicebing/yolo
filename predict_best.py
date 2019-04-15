import sys
import pandas as pd
import numpy as np

import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.utils.model_zoo as model_zoo
import torchvision
import torchvision.transforms as T

import load_dataset_v2
import yolo_loss_v2
import models_best

def transfer2normalratio(input_tensor):
    #input tensor = 1*7*7*26
    gx,gy = input_tensor.shape[1:3]
    out_tensor = input_tensor.clone()
    out_tensor[:,:,:,[0,1,0+(5+16),1+(5+16),0 +2*(5+16),1 +2*(5+16)]] *= 64
    out_tensor[:,:,:,[2,3,2+(5+16),3+(5+16),2 +2*(5+16),3 +2*(5+16)]] *= 448
    
    for i in range(gx):
        out_tensor[:,i,:,[0,0+(5+16),0 +2*(5+16)]] += 64*i
        out_tensor[:,:,i,[1,1+(5+16),1 +2*(5+16)]] += 64*i
    out_tensor = out_tensor.view(-1,21)
    return out_tensor

def Select_box(predict, C_threshold=0.001, IOU_threshold=0.5):
    #recover origin coord and flat
    predict = transfer2normalratio(predict)
    #split to class and box
    class_pred_tensor = predict[:,5:]
    boxes_pred = predict[:,:5]
    #predict class_num
    class_pred_value, class_pred_num = torch.max(class_pred_tensor,1)
    class_pred_value, class_pred_num = class_pred_value.float(), class_pred_num.float()

    #cal probability
    boxes_pred[:,4] *= class_pred_value
    #filter box1,box2
    mask_box1 = boxes_pred[:,4] > C_threshold
    #made box
    box_candidate = torch.cat([boxes_pred[mask_box1][:,:5],class_pred_num[mask_box1].view(-1,1)],1)
#    cand_box2 = torch.cat([boxes_pred[mask_box2][:,5:10],class_pred_num[mask_box2].view(-1,1)],1)    
#    box_candidate = torch.cat([cand_box1,cand_box2],0)
    
    #total class = 16
    out_box = torch.zeros([1,6]).to(device)
    for ic in range(16):
        mask_class = box_candidate[:,5] ==ic
        class_box =box_candidate[mask_class]
        #select box by NMS for each class
        box = NMS(class_box, IOU_T=IOU_threshold)
        if len(box)>0:
            for ibx, bx in enumerate(box):
                out_box = torch.cat([out_box,bx.view(1,-1)],0)
    #output x1,y1.x2.y2,class,prob
    result = torch.zeros([out_box.shape[0],10]).to(device)
    result[:,[0,1]] = torch.round(8/7*torch.clamp(out_box[:,[0,1]]-0.5*out_box[:,[2,3]],min=0,max=448))
    result[:,[4,5]] = torch.round(8/7*torch.clamp(out_box[:,[0,1]]+0.5*out_box[:,[2,3]],min=0,max=448))
    result[:,[2,3,6,7]] = result[:,[4,1,0,5]]
    result[:,[8,9]] = out_box[:,[5,4]]
    result_np = result[1:]#.detach().cpu().numpy()
    return box_candidate, result_np.detach().cpu().numpy()

def NMS(class_box, IOU_T=0.5):
    box_n=class_box.shape[0]
    outbox = []
    while torch.sum(class_box[:,4])>0:
        #select prob max one as select_box
        idx = torch.argmax(class_box[:,4])
        outbox.append(class_box[idx].clone())
        select_box = class_box[idx].repeat(box_n).view(box_n,-1)
        #got_iou
        select_iou = iou(class_box,select_box)
        #iou_filter
        mask_iou = select_iou > IOU_T
        #delete all > IOU_threshold including the selected box
        class_box[mask_iou] =0
    return outbox
    
def iou(box_pred,box_targ):
    #input size = [box_n,5]  target-1-to-all-predict
    box_n=box_targ.shape[0]
    #box_target num = box_n 
    #recover to normal size ratio
    area_pred = torch.prod(box_pred[:,[2,3]],1) #w*h
    area_targ = torch.prod(box_targ[:,[2,3]],1) #w*h
    Diff = torch.abs(box_pred-box_targ)
    Summ = (box_pred+box_targ)
    
    Status_x = Diff[:,0]>Diff[:,2]
    Status_y = Diff[:,1]>Diff[:,3]
    
    Ix = torch.zeros(box_n).float().to(device)
    Iy = torch.zeros(box_n).float().to(device)
    
    if (~Status_x).sum()>0:
        Ix[~Status_x]=(torch.min(box_pred[:,2],box_targ[:,2]))[~Status_x]
    if (~Status_y).sum()>0:
        Iy[~Status_y]=(torch.min(box_pred[:,3],box_targ[:,3]))[~Status_y]
    if (Status_x).sum()>0:
        Ix[Status_x]=(torch.clamp(0.5*Summ[:,2]-Diff[:,0],0))[Status_x]
    if (Status_y).sum()>0:
        Iy[Status_y]=(torch.clamp(0.5*Summ[:,3]-Diff[:,1],0))[Status_y]

    area_Inter = Ix*Iy
    out_IOU = area_Inter/(area_pred+area_targ-area_Inter+1e-6)

    return out_IOU    
    

#test 1 image
def test_img(model,C_T=0.001,I_T=0.5,outfilepath='./calculate/train_output_Tep400'):   
    model.to(device)
    criterion = yolo_loss_v2.YOLOloss_best(device).to(device)    
    model.eval()
    with torch.no_grad():
        data, target_old,target,fn = trainset[1]
        a = data.numpy()
        a =np.transpose(a,(1,2,0))
        plt.imshow(a)
        
        data, target = data.to(device), target.to(device)
        
        data = data.view(-1,3,448,448)
        predict = model(data)
        loss = criterion(predict.float(), target.float())
        box_candidate, out_box_np = Select_box(predict.float(),C_threshold=C_T,IOU_threshold=I_T)
        DF_output = pd.DataFrame(out_box_np)
        DF_output[8] = DF_output[8].map(DOTA_CLASSES)
        out_fn = os.path.join(outfilepath,str(fn[0])+'.txt')
        DF_output.to_csv(out_fn,sep=' ',header =False, index=False)
        print(fn,loss.item())
        return box_candidate, out_box_np, DF_output,fn,predict.float(),target.float()



        
def test_model_new(model,C_T=0.03,I_T=0.5,outfilepath='./calculate/train_output_Tep400'):   
    global total_loss
    model.to(device)  
    model.eval()
    
    iteration = 0
    for batch_idx, (data,fn) in enumerate(trainset_loader):
        data= data.to(device)
        predict = model(data)

        for i,pred in enumerate(predict):
            pred = pred.view(-1,7,7,63)
            box_candidate, out_box_np = Select_box(pred.float(),C_threshold=C_T,IOU_threshold=I_T)
            DF_output = pd.DataFrame(out_box_np)
            DF_output[8] = DF_output[8].map(DOTA_CLASSES)
            out_fn = os.path.join(outfilepath,str(fn[i])+'.txt')
            DF_output.to_csv(out_fn,sep=' ',header =False, index=False)
        print('Test [{}/{} ({:.0f}%)]'.format(
                batch_idx * len(data), len(trainset_loader.dataset),
                100. * batch_idx / len(trainset_loader)))
        iteration += 1


DOTA_CLASSES = {  # always index 0
    0:'plane', 1:'ship', 2:'storage-tank', 3:'baseball-diamond',
    4:'tennis-court', 5:'basketball-court', 6:'ground-track-field',
    7:'harbor', 8:'bridge', 9:'small-vehicle', 10:'large-vehicle',
    11:'helicopter', 12:'roundabout', 13:'soccer-ball-field',
    14:'swimming-pool', 15:'container-crane'}

try:
    filepath = sys.argv[1]
    outpath = sys.argv[2]
except:
    filepath = './hw2_train_val/val1500/'
    outpath = './output/'
    
device ='cuda'
batch = 10

yolo_net = models_best.Yolov1_vgg16bn(pretrained=False)
transform = T.Compose([T.ToTensor(),
                       T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
                      ])
    
trainset = load_dataset_v2.Dota_test(filepath,transform=transform,advanced=False,threshold = 0.001)
trainset_loader = DataLoader(trainset, batch_size=batch, shuffle=False, num_workers=10)

YOLO_BASE = torch.load('./model_zoo/BW_yolo_best.pth')

#box_candidate, out_box_np, DF_output,fn,predict,target = test_img(YOLO_BASE,
#                                                         C_T=0.01,
#                                                         I_T=0.5,
#                                                         outfilepath='./calculate')

import time
t0 = time.time()
test_model_new(YOLO_BASE, C_T=0.01,I_T=0.5, outfilepath=outpath)
print(time.time() - t0)
#print(total_loss/1500)
