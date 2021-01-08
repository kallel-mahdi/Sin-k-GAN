import torch
import os
from imageio import  imwrite
from cv2 import resize,imread
from singan import SinGAN
from log import TensorboardLogger

import argparse

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser(description='SinGAN - Random Sampling')
parser.add_argument('--run_name', required=True)
parser.add_argument('--img')
parser.add_argument('--N', type=int, default=8)
parser.add_argument('--steps_per_scale', type=int, default=2000)
parser.add_argument('--grad_penalty',type=bool,default=False)
parser.add_argument('--sink_steps',type=int,default=20)
parser.add_argument('--sink_eps',type=float,default=1)
parser.add_argument('--sink',type=bool,default=False)

args = parser.parse_args()

# get the available device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    
print("Gradient penalty =",args.grad_penalty)
print("Using sinkhorn loss = ",args.sink)

# instantiate the logger and the SinGAN
logger = TensorboardLogger(f'singan_{args.run_name}')
singan = SinGAN(N=args.N, logger=logger, device=device,grad_penalty=args.grad_penalty,sink=args.sink)


# load the single training image
train_img_path = os.path.join('data', args.img)
train_img = imread(train_img_path)
ref_img = imread("./data/car2_blue.jpg")
#ref_img = resize(ref_img,(train_img.shape[2],train_img.shape[1]))
ref_img = resize(ref_img,(train_img.shape[1],train_img.shape[0]))


print("Train_img shape",train_img.shape)
print("Ref_img shape",ref_img.shape)
#fit SinGan to one image 
## (ALWAYS PUT IMAGE IN A LIST)
singan.fit(img=[train_img], steps_per_scale=args.steps_per_scale)
# fit SinGAN to TWO IMAGES
#singan.fit(img=[train_img,ref_img], steps_per_scale=args.steps_per_scale)
# after training, save the model in a checkpoint
singan.save_checkpoint()
