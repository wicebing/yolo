import cv2
import os
import numpy as np
import pandas as pd
import glob
import random
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

DOTA_CLASSES = (  # always index 0
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'tennis-court', 'basketball-court', 'ground-track-field',
    'harbor', 'bridge', 'small-vehicle', 'large-vehicle',
    'helicopter', 'roundabout', 'soccer-ball-field',
    'swimming-pool', 'container-crane')

class Dota_tarin(Dataset):
    def __init__(self, root, transform=None, advanced=False, threshold = 0.1):
        """ Intialize the dataset """
        self.filenames = []
        self.root = root
        self.T = threshold
        self.transform = transform
        self.advanced = advanced
        # read filenames
        filenames = glob.glob(os.path.join(root,'images', '*.jpg'))
        for fn in filenames:
            img_num = fn.split('/')[-1].split('.')[0]
            label_fn = os.path.join(root,'labelTxt_hbb',img_num+'.txt')
            self.filenames.append((fn, label_fn))             
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn, label_fn = self.filenames[index]
#        image = Image.open(image_fn)
        img = cv2.imread(image_fn)
        gt_label = pd.read_csv(label_fn,header=None, sep=' ')
        gt_label[8]
        gt_label = np.array(gt_label)
        gt = gt_label[:,[0,1,4,5,8]]       #return x1,y1,x2,y2,class
        
        if self.advanced:
            img, gt = self.data_preprocess_advanced(img,gt)
        else:
            img, gt = self.data_preprocess(img,gt)
                
        Thes = self.T*np.random.rand(5)
#        img = self.random_bright(img,threshold=Thes[0])
        img = self.randomBlur(img,threshold=Thes[1])
        img = self.RandomBrightness(img,threshold=Thes[2])
        img = self.RandomHue(img,threshold=Thes[3])
        img = self.RandomSaturation(img,threshold=Thes[4])
        
        gtBox = self.GTtensor(gt)
        gtBox_b = self.GTtensor_best(gt)
        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(gtBox), torch.tensor(gtBox_b), image_fn.split('/')[4].split('.')[0] #gt

    def data_preprocess(self,img,gt):
        oH,oW = img.shape[:2]
        # step 1 rotation 90 ornot
        # step 2 flip
        # step 3 resize and translate
        NEW_S = int(448)
        img = cv2.resize(img,(NEW_S,NEW_S))
        gt[:,:4] = 7*gt[:,:4]/8      
        return img, gt

    def data_preprocess_advanced(self,img,gt):
        oH,oW = img.shape[:2]
        # step 1 rotation 90 ornot
        rotate_on = -90*random.randint(0,1)
        if rotate_on<0:
#            print('rotate=1')
            M =cv2.getRotationMatrix2D((256,256), rotate_on, 1.0)
            img = cv2.warpAffine(img, M, (oH,oW))
            gt[:,[0,1,2,3]] = gt[:,[1,0,3,2]]
            gt[:,[0,2]] = oW-1-gt[:,[0,2]]    
        #step 2 flip
        flip_type = random.randint(-1,2) #normal=2,3,4, flip = -1,0,1
        if flip_type<2:
#            print('Flip=',flip_type)
            img = cv2.flip(img,flip_type)
            if flip_type==1:
                gt[:,[0,2]] = oW-1-gt[:,[0,2]]
            elif flip_type==0:
                gt[:,[1,3]] = oH-1-gt[:,[1,3]]
            else:
                gt[:,[0,1,2,3]] = oW-1-gt[:,[0,1,2,3]]
        #step 3 resize and translate
        resize_range = random.randint(3,12) #normal=7,8,9, resize=5,6
        if resize_range > 6:
            NEW_S = int(448)
            img = cv2.resize(img,(NEW_S,NEW_S))
            gt[:,:4] = 7*gt[:,:4]/8
        else:
#            print('resize=',resize_range)
            NEW_S = int(resize_range*448/7)
            Padding_w = int((7-resize_range)*448/7)
            r_Pandding_w = random.randint(0,Padding_w)
            Padding_h = int((7-resize_range)*448/7)
            r_Pandding_h = random.randint(0,Padding_h)
            
            img = cv2.resize(img,(NEW_S,NEW_S))
            color = [0, 0, 0]
            img = cv2.copyMakeBorder(img,
                                        r_Pandding_w,
                                        Padding_w-r_Pandding_w,
                                        r_Pandding_h,
                                        Padding_h-r_Pandding_h,
                                        cv2.BORDER_CONSTANT,
                                        value=color)
            gt[:,:4] = resize_range*gt[:,:4]/8
            gt[:,[0,2]] = r_Pandding_h+gt[:,[0,2]]
            gt[:,[1,3]] = r_Pandding_w+gt[:,[1,3]]        
        return img, gt

    def GTtensor(self,gt):
        GTBox_tensor= np.zeros([7,7,26])
        for gtBox in gt:
            dx,dy,dw,dh =  0.5*(gtBox[0]+gtBox[2]),0.5*(gtBox[1]+gtBox[3]),abs(gtBox[2]-gtBox[0]),abs(gtBox[3]-gtBox[1])
            xi,yi,bx,by,bw,bh =int(dx/64),int(dy/64),(dx%64)/64,(dy%64)/64,dw/448,dh/448
            
            pattern = DOTA_CLASSES.index(gtBox[4])        
            GTBox_tensor[xi,yi,10+pattern] = 1
            if GTBox_tensor[xi,yi,4]==0:
                GTBox_tensor[xi,yi,0]= bx
                GTBox_tensor[xi,yi,1]= by
                GTBox_tensor[xi,yi,2]= bw
                GTBox_tensor[xi,yi,3]= bh
                GTBox_tensor[xi,yi,4]=1
                GTBox_tensor[xi,yi,5]= bx
                GTBox_tensor[xi,yi,6]= by
                GTBox_tensor[xi,yi,7]= bw
                GTBox_tensor[xi,yi,8]= bh
                GTBox_tensor[xi,yi,9]=1 
            else:
                GTBox_tensor[xi,yi,5]= bx
                GTBox_tensor[xi,yi,6]= by
                GTBox_tensor[xi,yi,7]= bw
                GTBox_tensor[xi,yi,8]= bh
                GTBox_tensor[xi,yi,9]=1
        return GTBox_tensor

    def GTtensor_best(self,gt):
        GTBox_tensor= np.zeros([7,7,3*(5+16)])  # 3 isolated bbox
        for gtBox in gt:
            dx,dy,dw,dh =  0.5*(gtBox[0]+gtBox[2]),0.5*(gtBox[1]+gtBox[3]),abs(gtBox[2]-gtBox[0]),abs(gtBox[3]-gtBox[1])
            xi,yi,bx,by,bw,bh =int(dx/64),int(dy/64),(dx%64)/64,(dy%64)/64,dw/448,dh/448
            
            pattern = DOTA_CLASSES.index(gtBox[4])        
            
            if GTBox_tensor[xi,yi,4]==0:
                GTBox_tensor[xi,yi,0]= bx
                GTBox_tensor[xi,yi,1]= by
                GTBox_tensor[xi,yi,2]= bw
                GTBox_tensor[xi,yi,3]= bh
                GTBox_tensor[xi,yi,4]=1
                GTBox_tensor[xi,yi,5+pattern] = 1
                
                GTBox_tensor[xi,yi,0+(5+16)]= bx
                GTBox_tensor[xi,yi,1+(5+16)]= by
                GTBox_tensor[xi,yi,2+(5+16)]= bw
                GTBox_tensor[xi,yi,3+(5+16)]= bh
                GTBox_tensor[xi,yi,4+(5+16)]=1
                GTBox_tensor[xi,yi,5+(5+16)+pattern] = 1

                GTBox_tensor[xi,yi,0 + 2*(5+16)]= bx
                GTBox_tensor[xi,yi,1 + 2*(5+16)]= by
                GTBox_tensor[xi,yi,2 + 2*(5+16)]= bw
                GTBox_tensor[xi,yi,3 + 2*(5+16)]= bh
                GTBox_tensor[xi,yi,4 + 2*(5+16)]=1
                GTBox_tensor[xi,yi,5 + 2*(5+16)+pattern] = 1
            elif GTBox_tensor[xi,yi,4+(5+16)]==0:
                GTBox_tensor[xi,yi,0+(5+16)]= bx
                GTBox_tensor[xi,yi,1+(5+16)]= by
                GTBox_tensor[xi,yi,2+(5+16)]= bw
                GTBox_tensor[xi,yi,3+(5+16)]= bh
                GTBox_tensor[xi,yi,4+(5+16)]=1
                GTBox_tensor[xi,yi,5+(5+16)+pattern] = 1
            else:
                GTBox_tensor[xi,yi,0 + 2*(5+16)]= bx
                GTBox_tensor[xi,yi,1 + 2*(5+16)]= by
                GTBox_tensor[xi,yi,2 + 2*(5+16)]= bw
                GTBox_tensor[xi,yi,3 + 2*(5+16)]= bh
                GTBox_tensor[xi,yi,4 + 2*(5+16)]=1
                GTBox_tensor[xi,yi,5 + 2*(5+16)+pattern] = 1              
        return GTBox_tensor


    def RandomBrightness(self,bgr,threshold = 0.1):
        if random.random() < threshold:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            v = v*adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr    
    def RandomSaturation(self,bgr,threshold = 0.1):
        if random.random() < threshold:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            s = s*adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr    
    def RandomHue(self,bgr,threshold = 0.1):
        if random.random() < threshold:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            h = h*adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr    
    def randomBlur(self,bgr,threshold = 0.1):
        if random.random()<threshold:
            bgr = cv2.blur(bgr,(5,5))
        return bgr    
    def random_bright(self, bgr, delta=16,threshold = 0.1):
        alpha = random.random()
        if alpha < threshold:
            bgr = bgr * alpha + random.randrange(-delta,delta)
            bgr = bgr.clip(min=0,max=255).astype(np.uint8)
        return bgr

    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    def BGR2HSV(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    def HSV2BGR(self,img):
        return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


class Dota_test(Dataset):
    def __init__(self, root, transform=None, advanced=False, threshold = 0.1):
        """ Intialize the dataset """
        self.filenames = []
        self.root = root
        self.T = threshold
        self.transform = transform
        self.advanced = advanced
        # read filenames
        filenames = glob.glob(os.path.join(root, '*.jpg'))
        for fn in filenames:
            self.filenames.append((fn,))             
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn, = self.filenames[index]
#        image = Image.open(image_fn)
        img = cv2.imread(image_fn)        

        NEW_S = int(448)
        img = cv2.resize(img,(NEW_S,NEW_S)) 
                        
        if self.transform is not None:
            img = self.transform(img)

        return img, image_fn.split('/')[-1].split('.')[0] #gt

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

def test():
    filepath = './hw2_train_val/train15000/'
    ds = Dota_tarin(filepath,threshold = 0.00)
    img, gtBox,fn = ds[3]
    print(img.shape, fn)
    plt.imshow(img)
#    print(gt)
#    import visualize_bbox
#    visualize_bbox.debug_plot(img,gt)
    return gtBox, fn
    
if __name__ == '__main__':
    gtBox,fn = test()