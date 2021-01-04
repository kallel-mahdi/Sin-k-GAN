import torch
import os
from imageio import imread, imwrite

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

args = parser.parse_args()

# get the available device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# instantiate the logger and the SinGAN
logger = TensorboardLogger(f'singan_{args.run_name}')
singan = SinGAN(N=args.N, logger=logger, device=device,grad_penalty=args.grad_penalty)


# load the single training image
train_img_path = os.path.join('data', args.img)
train_img = imread(train_img_path)
# fit SinGAN to it
singan.fit(img=train_img, steps_per_scale=args.steps_per_scale)
# after training, save the model in a checkpoint
singan.save_checkpoint()
