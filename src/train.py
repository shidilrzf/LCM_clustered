import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.optim
import torch.nn.functional as F
import torch.optim as optim
from data_loader import *
from models import *
from losses import *
from utils import *
import os


# Settings
use_cuda = False
use_cuda = use_cuda and torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")
latent_net_name = 'Latent4LSND'
clamp_images = True

num_cluster = 5

writer = SummaryWriter(log_dir='../runs')


# models
model_name = 'CelebA_cluster1'
laten_check_dir = '../' + model_name + '_latent/'

if not os.path.exists(laten_check_dir):
    os.makedirs(laten_check_dir)

# Generator model
generator = EncDecCelebA(in_channels=64)
generator.to(device)


# Datasets and loaders
batch_size = 10
num_train = 500
noise_sz = 40
dataset_folder = '/data/CelebA/celebA_redux_500/'




# Generate uniform noise
noise = uniform(1, 2, noise_sz, noise_sz)


# main train_validate loop
