from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBatchLeaky(nn.Conv2d):
    def __init__(self, lr_slope, *args, **kwargs):
        super(ConvBatchLeaky, self).__init__(*args, **kwargs)
        batch_dim = self.weight.data.size(0)
        self.bn = nn.BatchNorm2d(batch_dim)
        self.lr = nn.LeakyReLU(lr_slope)

    def forward(self, x):
        x = super(ConvBatchLeaky, self).forward(x)
        return self.lr(self.bn(x))


class ConvTrBatchLeaky(nn.ConvTranspose2d):
    def __init__(self, lr_slope, *args, **kwargs):
        super(ConvTrBatchLeaky, self).__init__(*args, **kwargs)
        batch_dim = self.weight.data.size(1)
        self.bn = nn.BatchNorm2d(batch_dim)
        self.lr = nn.LeakyReLU(lr_slope)

    def forward(self, x):
        x = super(ConvTrBatchLeaky, self).forward(x)
        return self.lr(self.bn(x))


class EncDecCelebA(nn.Module):

    def __init__(self, in_channels=1, lr_slope=0.2, bias=False):
        super(EncDecCelebA, self).__init__()
        self.lr_slope = lr_slope

        self.enc_conv1 = ConvBatchLeaky(self.lr_slope, in_channels, 256, 4, 2, 1, 1, bias=False)
        self.enc_conv2 = ConvBatchLeaky(self.lr_slope, 256, 512, 4, 2, 1, 1, bias=False)  # 8
        self.enc_conv3 = ConvBatchLeaky(self.lr_slope, 512, 1024, 4, 2, 1, 1, bias=False)  # 4
        self.enc_conv4 = ConvBatchLeaky(self.lr_slope, 1024, 1024, 3, 1, 2, 2, groups=512, bias=False)  # 4
        self.enc_conv5 = ConvBatchLeaky(self.lr_slope, 1024, 1024, 3, 1, 2, 2, groups=512, bias=False)  # 4
        self.enc_conv6 = ConvBatchLeaky(self.lr_slope, 1024, 1024, 3, 1, 2, 2, groups=512, bias=False)  # 4

        self.dec_conv1 = ConvTrBatchLeaky(0.2, 1024 + 1024, 512, 4, 2, 1, bias=bias)  # 8
        self.dec_conv2 = ConvTrBatchLeaky(0.2, 512 + 512, 512, 4, 2, 1, bias=bias)  # 16
        self.dec_conv3 = ConvTrBatchLeaky(0.2, 512, 256, 4, 2, 1, bias=bias)  # 32
        self.dec_conv4 = ConvTrBatchLeaky(0.2, 256, 128, 4, 2, 1, bias=bias)  # 64

        self.dec_conv5 = ConvBatchLeaky(0.2, 128, 64, 3, 1, 1, bias=bias)  # 128
        self.dec_conv6 = ConvBatchLeaky(0.2, 64, 32, 3, 1, 1, 1, bias=bias)  # 128

        self.dec_conv7 = nn.Conv2d(32, 3, 3, 1, 1, 1, bias=bias)
        self.upsamp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, input):
        #Encoder
        x1 = self.enc_conv1(input)
        x2 = self.enc_conv2(x1)
        x3 = self.enc_conv3(x2)

        x4 = self.enc_conv4(x3)
        x5 = self.enc_conv5(x4)
        x6 = self.enc_conv6(x5)

        #Decoder
        x = torch.cat([x6, x3], 1)
        x = self.dec_conv1(x)  # 8
        x = torch.cat([x, x2], 1)

        x = self.dec_conv2(x)  # 16
        x = self.dec_conv3(x)  # 32
        x = self.dec_conv4(x)  # 64

        x = self.upsamp(x)

        x = self.dec_conv5(x)  # 128
        x = self.dec_conv6(x)  # 128
        x = torch.sigmoid(self.dec_conv7(x))  # 128

        return x


class Latent4LSND(nn.Module):
    '''
    Latent model
    '''

    def __init__(self, lr_slope=0.2):
        super(Latent4LSND, self).__init__()
        self.lr_slope = lr_slope

        self.conv1 = ConvBatchLeaky(self.lr_slope, 2, 8, 3, 1, 0, 1, bias=False)  # 80
        self.conv2 = ConvBatchLeaky(self.lr_slope, 8, 16, 3, 1, 0, 1, bias=False)  # 76
        self.conv3 = ConvBatchLeaky(self.lr_slope, 16, 32, 3, 1, 0, 1, bias=False)  # 72

        self.conv4 = nn.Conv2d(32, 64, 3, 1, 0, 1, bias=False)  # 68

    def restrict(self, min_val, max_val):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                for p in m.parameters():
                    p.data.clamp_(min_val, max_val)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.leaky_relu(self.conv4(x), self.lr_slope)

        return x
