import pandas as pd
import numpy as np
from PIL import Image
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



__all__ = ['vgg16_bn']
model_urls = {
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}

DOTA_CLASSES = (  # always index 0
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'tennis-court', 'basketball-court', 'ground-track-field',
    'harbor', 'bridge', 'small-vehicle', 'large-vehicle',
    'helicopter', 'roundabout', 'soccer-ball-field',
    'swimming-pool', 'container-crane')


class VGG(nn.Module):

    def __init__(self, features, output_size=1274, image_size=448):
        super(VGG, self).__init__()
        self.features = features
        self.image_size = image_size

        self.rc1 = nn.Sequential(nn.Conv2d(512,1024,1),
                                 nn.BatchNorm2d(1024),
                                 nn.LeakyReLU(1e-1),
                                 nn.Dropout2d(0.3),
                                 nn.Conv2d(1024,512,3,padding=1),
                                 nn.BatchNorm2d(512),
                                 nn.LeakyReLU(1e-1),
                                 nn.Dropout(0.5),                                  
                                 )
        self.rc2 = nn.Sequential(nn.Conv2d(1024,2048,1),
                                 nn.BatchNorm2d(2048),
                                 nn.LeakyReLU(1e-1),
                                 nn.Dropout2d(0.3),
                                 nn.Conv2d(2048,1024,3,padding=1),
                                 nn.BatchNorm2d(1024),
                                 nn.LeakyReLU(1e-1),
                                 nn.Dropout(0.5),
                                 )
        self.rc3 = nn.Sequential(nn.Conv2d(2048,4096,1),
                                 nn.BatchNorm2d(4096),
                                 nn.LeakyReLU(1e-1),
                                 nn.Dropout2d(0.3),
                                 nn.Conv2d(4096,2048,3,padding=1),
                                 nn.BatchNorm2d(2048),
                                 nn.LeakyReLU(1e-1),
                                 nn.Dropout(0.5),
                                 )
        self.out = nn.Sequential(nn.Conv2d(4096,2048,1),
                                 nn.BatchNorm2d(2048),
                                 nn.LeakyReLU(1e-1),
                                 nn.Dropout2d(0.3),
                                 nn.Conv2d(2048,1024,1),
                                 nn.BatchNorm2d(1024),
                                 nn.LeakyReLU(1e-1),
                                 nn.Dropout2d(0.3),
                                 nn.Conv2d(1024,512,1),
                                 nn.BatchNorm2d(512),
                                 nn.LeakyReLU(1e-1),
                                 nn.Dropout2d(0.3),
                                 nn.Conv2d(512,63,1),
                                 nn.Sigmoid(),
                                 )
        
        self._initialize_weights()

    def forward(self, x):
        x0 = self.features(x)
        x1 = self.rc1(x0)
        x = torch.cat([x0,x1],1)
        x2 = self.rc2(x)
        x = torch.cat([x0,x1,x2],1)
        x3 = self.rc3(x)
        x = torch.cat([x0,x1,x2,x3],1)
        x = self.out(x)

        x = torch.transpose(x,1,3)
        x = x.contiguous()
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    s = 1
    first_flag=True
    for v in cfg:
        s=1
        if (v==64 and first_flag):
            s=2
            first_flag=False
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=s, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def conv_bn_relu(in_channels,out_channels,kernel_size=3,stride=2,padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )

cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

def Yolov1_vgg16bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    yolo = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        vgg_state_dict = model_zoo.load_url(model_urls['vgg16_bn'])
        yolo_state_dict = yolo.state_dict()
        for k in vgg_state_dict.keys():
            if k in yolo_state_dict.keys() and k.startswith('features'):
                yolo_state_dict[k] = vgg_state_dict[k]
        yolo.load_state_dict(yolo_state_dict)
    return yolo


def test():
    import torch
    model = Yolov1_vgg16bn(pretrained=True)
    img = torch.rand(1,3,448,448)
    output = model(img)
    print(output.size())

if __name__ == '__main__':
    test()

