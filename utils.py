import torch
import torch.nn as nn
import numpy as np
from skimage.transform import resize
from skimage.io import imread
import torchvision.utils as tvu
import cv2


def nan_check_and_break(tensor, name=""):
    if isinstance(input, list) or isinstance(input, tuple):
        for tensor in input:
            return(nan_check_and_break(tensor, name))
    else:
        if nan_check(tensor, name) is True:
            exit(-1)


def nan_check(tensor, name=""):
    if isinstance(input, list) or isinstance(input, tuple):
        for tensor in input:
            return(nan_check(tensor, name))
    else:
        if torch.sum(torch.isnan(tensor)) > 0:
            print("Tensor {} with shape {} was NaN.".format(name, tensor.shape))
            return True

        elif torch.sum(torch.isinf(tensor)) > 0:
            print("Tensor {} with shape {} was Inf.".format(name, tensor.shape))
            return True

    return False


def zero_check_and_break(tensor, name=""):
    if torch.sum(tensor == 0).item() > 0:
        print("tensor {} of {} dim contained ZERO!!".format(name, tensor.shape))
        exit(-1)


def all_zero_check_and_break(tensor, name=""):
    if torch.sum(tensor == 0).item() == np.prod(list(tensor.shape)):
        print("tensor {} of {} dim was all zero".format(name, tensor.shape))
        exit(-1)


def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight.data)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Sequential):
            for sub_mod in m:
                init_weights(sub_mod)


def sample_uniform(shape, a=-1, b=1, var=1. / 10, use_cuda=False):
    shape = list(shape) if isinstance(shape, tuple) else shape
    type_tfloat = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    return type_tfloat(*shape).uniform_(a, b) * var


def visualize_clusters(img_list, labels, im_size=128, num_cluster=10):
    labels = np.asarray(labels)
    img_grid = np.zeros((im_size * num_cluster, im_size * 5, 3))
    for cluster in range(num_cluster):
        ind = np.where(labels == cluster)[0]
        if ind.shape[0] == 0:
            continue
        for j in range(min(ind.shape[0], 5)):
            img = imread(img_list[ind[j]])
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = resize(img, (im_size, im_size), mode='constant')
            img_grid[cluster * im_size:cluster * im_size + im_size, j * im_size:j * im_size + im_size, :] = img

    img_grid *= 255
    return img_grid.astype(np.uint8).transpose(2, 0, 1)
