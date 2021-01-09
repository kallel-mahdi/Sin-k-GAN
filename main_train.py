import torch
import os
from imageio import imread, imwrite
from cv2 import resize

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
parser.add_argument('--g_lr',type=float,default= 5e-4)  # learning rate for generators
parser.add_argument('--d_lr',type=float,default= 5e-4),  # learning rate for discriminators
parser.add_argument('--n_blocks',type=float,default=5)  # number of convblocks in each of the N generators (modules)
parser.add_argument('--base_n_channels',type=float,default= 32)  # base number of filters for the coarsest module
parser.add_argument('--min_n_channels',type=float,default= 32) # minimum number of filters in any layer of any module
parser.add_argument('--rec_loss_weight',type= float,default=10.0)  # alpha weight for reconstruction loss
parser.add_argument('--grad_penalty_weight',type=float, default=0.1)  # lambda weight for gradient penalty loss
parser.add_argument('--noise_weight',type=float, default=0.1)  # base standard deviation of gaussian noise

args = parser.parse_args()

# get the available device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    
print("Gradient penalty =",args.grad_penalty)
print("Using sinkhorn loss = ",args.sink)
# hypers 
hypers = {
            'n_blocks': args.n_blocks,  
            'base_n_channels': args.base_n_channels,
            'min_n_channels': args.min_n_channels, 
            'rec_loss_weight': args.rec_loss_weight,  
            'grad_penalty_weight': args.grad_penalty_weight,
            'noise_weight': args.noise_weight,
         }
# instantiate the logger and the SinGAN
logger = TensorboardLogger(f'singan_{args.run_name}')
singan = SinGAN(N=args.N, logger=logger, device=device,grad_penalty=args.grad_penalty,sink=args.sink,hypers=hypers)


# load the single training image
train_img_path = os.path.join('data', args.img)
train_img = imread(train_img_path)

# fit SinGAN to it
## Always put image in a list
singan.fit(img=[train_img], steps_per_scale=args.steps_per_scale)
# after training, save the model in a checkpoint
singan.save_checkpoint()
