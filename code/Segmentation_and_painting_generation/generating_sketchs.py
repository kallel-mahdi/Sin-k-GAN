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
import matplotlib.pyplot as plt
import random

#==============================
import model
import preprocessing


def pil_loader(path):
      with open(path, 'rb') as f:
          with Image.open(f) as img:
              return img.convert('RGB')

def get_sketch(path_image,model,transform_m,pretrained,transform_p):
    image = pil_loader(path_image)
    tensor_im = transform_p(image)[None,:,:,:]
    mask_car = pretrained(tensor_im)['out'][0].argmax(0)
    mask_car[mask_car>0] = 1
    image = np.array(image)
    img_crp = np.zeros_like(image)
    img_crp[np.where(mask_car>0)[0],np.where(mask_car>0)[1],:] = image[np.where(mask_car>0)[0],np.where(mask_car>0)[1],:]
    tensor_im = transform_m(image)[None,:,:,:]
    out = torch.argmax(model(tensor_im),dim=1).cpu()[0].numpy()
    return resize(out, (image.shape[0], image.shape[1]), order=0, preserve_range=True, anti_aliasing=False).astype('uint8')



def get_all_sketchs(path_test_images,path_train_image,model,transform_m,pretrained,transform_p,labels_dict):
    dic_colors_train = {}
    sketchs = []
    orientation_train = 'right'
    sketch_image_train = get_sketch(path_train_image,model,transform_m,pretrained,transform_p)    
    labels_train = np.unique(sketch_image_train)
    image_train = np.array(pil_loader(path_train_image))

    # 7 label of head lights
    # 6 label of wheels
    wheel_ind = np.mean(np.where(sketch_image_train==6)[1])
    lights_ind = np.mean(np.where(sketch_image_train==7)[1])
    if wheel_ind > lights_ind:
      orientation_train = 'left'
    for l in labels_train:    
      mean = np.mean(random.choices(image_train[sketch_image_train==l], k = 10000),axis = 0)
      dic_colors_train[l] = mean.astype(int)
    dic_colors_train[l] =[0,0,0]
    del image_train
    del sketch_image_train

    for i, path in enumerate(path_test_images):
      orientation_test = 'right'
      sketch = get_sketch(path,model,transform_m,pretrained,transform_p)
      wheel_ind = np.mean(np.where(sketch==6)[1])
      lights_ind = np.mean(np.where(sketch==7)[1])
      if wheel_ind > lights_ind:
        print('in')
        orientation_test = 'left'
      color_sketch = np.zeros((sketch.shape[0],sketch.shape[1],3)).astype(int)
      for l in labels_train :
        if l in dic_colors_train.keys():
          color_sketch[sketch==l]=dic_colors_train[l]
      

      if orientation_test != orientation_train:
        color_sketch = cv2.flip(color_sketch, 1)

      color_sketch= cv2.resize(color_sketch.astype('float32') , dsize=(5*color_sketch.shape[1],5*color_sketch.shape[0]), interpolation=cv2.INTER_NEAREST)
      color_sketch = Image.fromarray(np.uint8(color_sketch))


      plt.imshow(color_sketch)
      plt.show()
      try:
        color_sketch.save(f'{main_path}/sketchs/{os.path.basename(path_train_image)[:-4]}/{os.path.basename(path_train_image)[:-4]}_sketch_{os.path.basename(path)}')
      except OSError:
        try:
          os.makedirs(f'{main_path}/sketchs/{os.path.basename(path_train_image)[:-4]}/')
          color_sketch.save(f'{main_path}/sketchs/{os.path.basename(path_train_image)[:-4]}/{os.path.basename(path_train_image)[:-4]}_sketch_{os.path.basename(path)}')
        except Exception:
          print('Folder not created')
      #sketchs.append(color_sketch)



# transform_m = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((300,300)),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    
# ])

# transform_p = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])


# dic = preprocessing.labels_car_parts()

# pretrained = torch.hub.load('pytorch/vision:v0.6.0', 'fcn_resnet101', pretrained=True)
# pretrained.eval()
# # ct = 0
# # for child in pretrained.children():
# #   ct += 1
# #   if 0<ct<2:
# #     for param in child.parameters():
# #         param.requires_grad = False

# model = my_FCN(torch.hub.load('pytorch/vision:v0.6.0', 'fcn_resnet101', pretrained=True))
# model.load_state_dict(torch.load('/content/drive/My Drive/SinGAN-Project/training_data_seg/model_seg_aug_free_exact_mask_3.pth'))
# model.eval()



# path_train = f'{main_path}/train_images/'
# path_test = f'{main_path}/test_images/'
# path_test_images = [path_test+f for f in os.listdir(path_test)]
# for image in os.listdir(path_train):
#   path_train_image = path_train+image
#   get_all_sketchs(path_test_images,path_train_image,model,transform_m,pretrained,transform_p,dic)
