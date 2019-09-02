import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# def gauss_kernel(k_size=5, sigma=1.0, use_cuda=False):
#     if k_size % 2 != 1:
#         raise ValueError("kernel size must be uneven")

#     x = torch.linspace(- k_size / 2, k_size / 2, k_size)
#     grid = torch.stack([x.repeat(k_size, 1).t().contiguous().view(-1), x.repeat(k_size)], 1)

#     distsq = torch.pow(grid[:, 0], 2) + torch.pow(grid[:, 1], 2)
#     denom = torch.pow(torch.Tensor([sigma]), 2) * 2.0
#     kernel = torch.exp(- distsq / denom)
#     kernel /= torch.sum(kernel)
#     kernel = torch.reshape(kernel, (k_size, k_size))
#     kernel = kernel.cuda() if use_cuda else kernel
#     return kernel


def dog_pyramid(im, n_levels=5, k_size=15, sigma=4.0, use_cuda=False):

    kernel = gauss_kernel(k_size, sigma, use_cuda)
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    func_pad = nn.ReplicationPad2d((k_size // 2, k_size // 2, k_size // 2, k_size // 2))
    curr_im = im
    n_levels = 5
    log_pyr = []
    for i in range(n_levels):
        conv_im = F.conv2d(func_pad(curr_im), kernel)
        diff = curr_im - conv_im
        log_pyr.append(diff)
        curr_im = F.avg_pool2d(conv_im, 2)

    return log_pyr


def loss_dog(im1, im2, n_levels=5, k_size=5, sigma=2.0, use_cuda=False):
    pyramid_im1 = dog_pyramid(im1, n_levels, k_size, sigma, use_cuda)
    pyramid_im2 = dog_pyramid(im2, n_levels, k_size, sigma, use_cuda)
    return sum(F.l1_loss(a, b) for a, b in zip(pyramid_im1, pyramid_im2)) / n_levels


def gaussian(x, size, sigma):
    return np.exp((x - size // 2)**2 / (-2 * sigma**2))**2


def gauss_kernel(size=5, sigma=1.0):
    '''
    Lap Loss adapted from: https://github.com/mtyka/laploss/blob/master/laploss.py
    '''
    grid = np.float32(np.mgrid[0:size, 0:size].T)
    kernel = np.sum(gaussian(grid, size, sigma), axis=2)
    kernel /= np.sum(kernel)
    return kernel


def conv_gauss(t_input, stride=1, k_size=5, sigma=1.6, repeats=1):
    t_kernel_np = gauss_kernel(size=k_size, sigma=sigma).reshape([1, 1, k_size, k_size])
    t_input_device = t_input.device
    t_kernel = torch.from_numpy(t_kernel_np)
    num_channels = t_input.data.shape[1]
    t_kernel3 = torch.cat([t_kernel] * num_channels, 0).type(torch.FloatTensor).to(t_input_device)
    t_result = t_input.type(torch.FloatTensor).to(t_input_device)

    for r in range(repeats):
        t_result = F.conv2d(t_result, t_kernel3, stride=1, padding=2, groups=num_channels)
    return t_result


def make_laplacian_pyramid(t_img, max_levels):
    t_pyr = []
    current = t_img.type(torch.FloatTensor)
    for level in range(max_levels):
        t_gauss = conv_gauss(current, stride=1, k_size=5, sigma=2.0)
        t_diff = current - t_gauss
        t_pyr.append(t_diff)
        current = F.avg_pool2d(t_gauss, 2, 2)
        t_pyr.append(current)
    return t_pyr


def laploss(t_img1, t_img2, max_levels=3):
    t_pyr1 = make_laplacian_pyramid(t_img1, max_levels)
    t_pyr2 = make_laplacian_pyramid(t_img2, max_levels)
    loss = 0.0
    for i in range(len(t_pyr1)):
        loss += (2**(-2 * i)) * L1_loss(t_pyr1[i], t_pyr2[i])
    return loss


def L1_loss(inputs, targets):
    # print(inputs.size())
    # print(targets.size())
    return (inputs - targets).abs().mean()
