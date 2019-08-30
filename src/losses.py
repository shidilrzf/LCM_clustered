import torch
import torch.nn as nn
import torch.nn.functional as F


def gauss_kernel(k_size=5, sigma=1.0, use_cuda=False):
    if k_size % 2 != 1:
        raise ValueError("kernel size must be uneven")

    x = torch.linspace(- k_size / 2, k_size / 2, k_size)
    grid = torch.stack([x.repeat(k_size, 1).t().contiguous().view(-1), x.repeat(k_size)], 1)

    distsq = torch.pow(grid[:, 0], 2) + torch.pow(grid[:, 1], 2)
    denom = torch.pow(torch.Tensor([sigma]), 2) * 2.0
    kernel = torch.exp(- distsq / denom)
    kernel /= torch.sum(kernel)
    kernel = torch.reshape(kernel, (k_size, k_size))
    kernel = kernel.cuda() if use_cuda else kernel
    return kernel


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
