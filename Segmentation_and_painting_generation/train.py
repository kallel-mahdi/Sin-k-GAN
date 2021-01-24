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
#==============================
import model
import preprocessing


def accuracy(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()

def train(epoch,criterion,accuracy):
    model.train()
    running_acc = 0
    for batch_idx, (data, target) in enumerate(train_loader):
      if use_cuda:
          data, target = data.cuda(), target.cuda()
          target =torch.squeeze(target, 1).cuda()
      optimizer.zero_grad()
      output = model(data)
      loss = criterion(F.log_softmax(output), target.long())
      loss.backward()
      optimizer.step()
      acc = accuracy(output, target.long())
      running_acc += acc*train_loader.batch_size

      if batch_idx % 10 == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {}'.format(
              epoch, batch_idx * len(data), len(train_loader.dataset),
              100. * batch_idx / len(train_loader), loss.data.item(), acc))
    running_acc /= len(train_loader.dataset)
    print('\nTrain accuracy:{:.4f}'.format(running_acc))
    
            

best_acc = 0
j = 0

def validation(epoch,accuracy):
    model.eval()
    validation_loss = 0
    correct = 0
    running_acc = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
            target =torch.squeeze(target, 1).cuda()
        
        output = model(data)
        acc = accuracy(output,target.long())
        running_acc += acc*val_loader.batch_size

    
    
    running_acc /= len(val_loader.dataset)
    print('\nValidation accuracy:{:.4f}'.format(running_acc))
    plt.imshow(torch.argmax(model(image_preprocess(pil_loader(main_path+'/test_images/bmw.jpg'))[None,:,:,:].cuda()),dim=1).cpu()[0].numpy())
    plt.show()

    global best_acc 
    global j


    if best_acc < running_acc:
        j=1
        best_acc = running_acc
        print('\nBest accuracy:{:.4f}'.format(best_acc))
        torch.save(model.state_dict(), main_path+'training_data_seg/model_seg_aug_free_exact_mask_3.pth')
        print('model saved')
    else:
      if j==5:  
        j=1  
        scheduler.step()
        for param_group in optimizer.param_groups:
                print("Current learning rate is: {}".format(param_group['lr']))      


if use_cuda:
  print('Using GPU')
  model.cuda()
else:
  print('Using CPU')



# subsets = train_val_dataset(my_dataset, val_split=0.1)
# train_loader = DataLoader(Custom_Dataset(folder_path = '/content/drive/My Drive/training_data_seg/',transform=[image_preprocess,mask_preprocess]),batch_size = 10, shuffle=True)

# train_loader = torch.utils.data.DataLoader(subsets['train'],batch_size=16, shuffle=True, num_workers=1)
# val_loader = torch.utils.data.DataLoader(subsets['val'] ,batch_size=2, shuffle=False, num_workers=1)
# criterion = nn.NLLLoss2d(weight = torch.tensor([1.,1.,1.,1.,1.,1.,1.5,4.,1.], requires_grad=False).cuda())
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
# pretrained = torch.hub.load('pytorch/vision:v0.6.0', 'fcn_resnet101', pretrained=True)
# # ct = 0
# # for child in pretrained.children():
# #   ct += 1
# #   if 0<ct<3:
# #     for param in child.parameters():
# #         param.requires_grad = False

# model = my_FCN(pretrained)
# use_cuda=True
# model.load_state_dict(torch.load('/content/drive/My Drive/training_data_seg/model_seg_free_aug_decoder.pth'))

# for epoch in range(50):
#   train(epoch,criterion,accuracy)
#   validation(epoch,accuracy)
