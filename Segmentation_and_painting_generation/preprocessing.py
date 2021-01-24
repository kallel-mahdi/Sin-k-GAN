import numpy as np 
import os
from sklearn.model_selection import train_test_split
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.functional as F
import zipfile
import torchvision.transforms as transforms
import PIL.Image as Image
import torchvision.models as models
import skimage.io as io
from torch.utils.data import TensorDataset, DataLoader,Dataset
from PIL import Image
import torchvision
import cv2
from skimage.transform import resize
import natsort
from functools import partial
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
! git clone https://github.com/twuilliam/pascal-part-py # This repository helped us for preprocessing PASCAL-Parts Dataset (2010)  
%cd pascal-part-py 
from anno import ImageAnnotation


class Custom_Dataset(Dataset):
    def __init__(self, folder_path , transform):
        super(Custom_Dataset, self).__init__()
        self.img_files =[f'{folder_path}/cars_images/{image}' for image in  os.listdir(f'{folder_path}/cars_images/')]
        self.mask_files = []
        for img in self.img_files:
             self.mask_files.append(os.path.join(folder_path,'masks/','m'+os.path.basename(img)[:-4]+'.txt')) 
        self.transform = transform

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            img = self.transform[0](pil_loader(img_path))
            mask = self.transform[1](np.loadtxt(mask_path))
            return img, mask

    def __len__(self):
        return len(self.img_files)

#=================================================================================================#

def clean_dataset(path_dataset):

  """Cleaning PASCAL-Parts Dataset"""

  dir_images = [f[:-4] for f in os.listdir(path_dataset+'/JPEGImages')]
  dir_annotations = [f[:-4] for f in os.listdir(path_dataset+'/Annotations')]
  for f in dir_images:
    if f not in dir_annotations:
      os.remove(path_dataset+'/JPEGImages/'+f+'.jpg')
  for f in dir_annotations:
    if f not in dir_images:
      os.remove(path_dataset+'/Annotations/'+f+'.mat')


def select_car_images(data,path_dataset,main_path):

  """Create cars Dataset"""

  an = ImageAnnotation(main_path+'VOC/JPEGImages/'+data+'.jpg',main_path+'/Annotations/'+data+'.mat')
  objects = [an.objects[i].class_name for i in range(len(an.objects))]
  if 'car' in objects:
    img = pil_loader(path_dataset+'/JPEGImages/'+data+'.jpg')
    img.save(f'{main_path}/training_data_seg/cars_images/{data}.jpg')
    img_mask = transforms.ToPILImage()(an.part_mask)
    img_mask.save(f'{main_path}/training_data_seg/cars_masks/m{data}.jpg')




def labels_car_parts():
  dic = {}
  dic['backround']   = 0
  dic['unk'] = 1
  dic['frontside']   = 2
  dic['leftside']    = 2
  dic['rightside']   = 2
  dic['backside']    = 2
  dic['roofside']    = 2
  dic['leftmirror']  = 3
  dic['rightmirror'] = 3   
  dic['fliplate']    = 4 
  dic['bliplate']    = 4     
  for ii in range(1, 10 + 1):
      dic[('door_%d' % ii)] = 5
  for ii in range(1, 10 + 1):
      dic[('wheel_%d' % ii)] = 6
  for ii in range(1, 10 + 1):
      dic[('headlight_%d' % ii)] = 7
  for ii in range(1, 20 + 1):
      dic[('window_%d' % ii)] = 8
  return dic



def car_mask_bw(path_images,main_path):

  """ Creating customized masks for car's parts """

   lab = labels_car_parts()
   files = os.listdir(path_images)
   for f in files:
     an = ImageAnnotation(main_path+'/JPEGImages/'+f[:-4]+'.jpg',main_path+'/Annotations/'+f[:-4]+'.mat')
     objects_names = [an.objects[i].class_name for i in range(len(an.objects))]
     cars_instences = np.where(np.array(objects_names) =='car')[0]
     out = np.zeros(an.imsize[:-1])
     for i in cars_instences:
        idx_rw = np.where(an.objects[i].mask>0)[0]
        idx_col = np.where(an.objects[i].mask>0)[1]
        out[idx_rw,idx_col] = lab['unk']
        parts = an.objects[i].parts
        for part in parts:
            mask = part.mask
            idx_rw = np.where(mask>0)[0]
            idx_col = np.where(mask>0)[1]
            out[idx_rw,idx_col] = lab[part.part_name]
     np.savetxt(f'{main_path}/training_data_seg/masks/m{f[:-4]}.txt',out, fmt="%s")



def data_augmentation(main_path):

  """ Horizontal flip and masking backgound"""

  for f in tqdm(os.listdir(main_path+'/training_data_seg/cars_images')):
      if 'jpg' in f:
          txt = np.loadtxt(main_path+'/training_data_seg/masks/m'+ f[:-4]+'.txt')
          img = np.array(pil_loader('/content/drive/My Drive/training_data_seg/cars_images/'+ f))
          img[np.where(txt==0)[0],np.where(txt==0)[1],:] = [0,0,0] # preserving only car in the image
          img2 = cv2.flip(img, 1) # horizontal flip
          img = Image.fromarray(np.uint8(img))
          img2 = Image.fromarray(np.uint8(img2))
          img.save(main_path+'/training_data_seg/cars_images_aug_masked/'+ f)
          img2.save(main_path+'/training_data_seg/cars_images_aug_masked/flip'+ f)
         

  for f in tqdm(os.listdir(main_path+'/training_data_seg/masks')):
      if 'txt' in f:
          txt = np.loadtxt(main_path+'/training_data_seg/masks/'+ f)
          txt2 = cv2.flip(txt, 1)
          np.savetxt(f'{main_path}/training_data_seg/masks_aug/m{f[1:-4]}.txt',txt, fmt="%s")
          np.savetxt(f'{main_path}/training_data_seg/masks_aug/mflip{f[1:-4]}.txt',txt2, fmt="%s")


def hot_encoding(mask,num_classes=9):
  mask = np.squeeze(mask.numpy().astype(int))
  m = np.zeros((9,mask.shape[0],mask.shape[1]))
  for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
      m[mask[i,j],i,j] = 1
  return torch.from_numpy(m)
      
def train_val_dataset(dataset, val_split=0.1):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split , shuffle= True)
    sets = {}
    sets['train'] = torch.utils.data.Subset(dataset, train_idx)
    sets['val'] = torch.utils.data.Subset(dataset, val_idx)
    return sets

image_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((300,300)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    
])
mask_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((300,300)),
    torch.round,
    #hot_encoding,       
    ])