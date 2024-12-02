# -*- coding: utf-8 -*-
# @Time    : 2021/7/21 16:48
# @Author  : Ran.Gu
# @Email   : guran924@std.uestc.edu.cn
'''
cs-cada uses different normalization.
'''
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
from torch import nn


class _DomainSpecificBatchNorm(nn.Module):
    _version = 2

    def __init__(self, num_features, num_classes, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_DomainSpecificBatchNorm, self).__init__()
        #         self.bns = nn.ModuleList([nn.modules.batchnorm._BatchNorm(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, domain_label):
        self._check_input_dim(x)
        bn = self.bns[domain_label]
        return bn(x)


class DomainSpecificBatchNorm2d(_DomainSpecificBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class Unet_dsbn_cont(nn.Module):
    def __init__(self, net_params:dict):
        super(Unet_dsbn_cont, self).__init__()
        self.num_filters = net_params['num_filters']
        self.num_channels = net_params['num_channels']
        self.num_classes = net_params['num_classes']
        self.num_classes_t = net_params['num_classes_t']
        self.normalization = net_params['normalization']
        self.num_domain = net_params['num_domains']
        filters = [self.num_filters,
                   self.num_filters * 2,
                   self.num_filters * 4,
                   self.num_filters * 8,
                   self.num_filters * 16]

        self.conv1 = conv_block(self.num_channels, filters[0], normalization=self.normalization, num_domain=self.num_domain)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = conv_block(filters[0], filters[1], normalization=self.normalization, num_domain=self.num_domain)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = conv_block(filters[1], filters[2], normalization=self.normalization, num_domain=self.num_domain)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = conv_block(filters[2], filters[3], drop_out=True, normalization=self.normalization, num_domain=self.num_domain)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.center = conv_block(filters[3], filters[4], drop_out=True, normalization=self.normalization, num_domain=self.num_domain)

        # f1 and g1 encoder
        self.f1 = nn.Sequential(nn.Conv2d(filters[4], 64, kernel_size=3, padding=1, bias=True),
                                nn.Conv2d(64, 16, kernel_size=1))
        self.g1 = nn.Sequential(nn.Linear(in_features=16384, out_features=4096),
                                nn.ReLU(),
                                nn.Linear(in_features=4096, out_features=1024),
                                nn.ReLU(),
                                nn.Linear(in_features=1024, out_features=512))

        # upsample
        self.up4 = UpCatconv(filters[4], filters[3], drop_out=True, normalization=self.normalization, num_domain=self.num_domain)
        self.up3 = UpCatconv(filters[3], filters[2], normalization=self.normalization, num_domain=self.num_domain)
        self.up2 = UpCatconv(filters[2], filters[1], normalization=self.normalization, num_domain=self.num_domain)
        self.up1 = UpCatconv(filters[1], filters[0], normalization=self.normalization, num_domain=self.num_domain)
        
        self.final = nn.Sequential(nn.Conv2d(filters[0], filters[0], kernel_size=1),
                                   nn.Conv2d(filters[0], self.num_classes, kernel_size=1))
        self.final2 = nn.Sequential(nn.Conv2d(filters[0], filters[0], kernel_size=1),
                                   nn.Conv2d(filters[0], self.num_classes_t, kernel_size=1))

    def forward(self, x, domain_label,is_td):
        # print("tensor",x.shape)
        conv1 = self.conv1(x, domain_label)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1, domain_label)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2, domain_label)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3, domain_label)
        pool4 = self.pool4(conv4)

        center = self.center(pool4, domain_label)
        high_d = self.f1(center)
        high_d_represent = self.g1(high_d.reshape(high_d.size(0), -1))

        up_4 = self.up4(conv4, center, domain_label)
        up_3 = self.up3(conv3, up_4, domain_label)
        up_2 = self.up2(conv2, up_3, domain_label)
        up_1 = self.up1(conv1, up_2, domain_label)

        if is_td:
            out = self.final2(up_1)
        else:
            out = self.final(up_1)
        return out, high_d_represent


class DropBlock(nn.Module):
    def __init__(self, block_size: int, p: float = 0.5):
        super().__init__()
        self.block_size = block_size
        self.p = p


    def calculate_gamma(self, x):
        """Compute gamma, eq (1) in the paper
        Args:
            x (Tensor): Input tensor
        Returns:
            Tensor: gamma
        """
        
        invalid = (1 - self.p) / (self.block_size ** 2)
        valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return invalid * valid



    def forward(self, x):
        if self.training:
            gamma = self.calculate_gamma(x)
            mask = torch.bernoulli(torch.ones_like(x) * gamma)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x


# conv_block(nn.Module) for U-net convolution block
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, drop_out=False, normalization='none', num_domain = 6):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=True)
        self.normalization = normalization
        if normalization == 'batchnorm':
            self.bn = nn.BatchNorm2d(ch_out)
        elif normalization == 'instancenorm':
            self.bn = nn.InstanceNorm2d(ch_out)
        elif normalization == 'dsbn':
            self.bn = DomainSpecificBatchNorm2d(ch_out, num_domain)
        elif normalization != 'none':
            assert False
        self.relu = nn.ReLU(inplace=True)
        self.dropout = drop_out

    def forward(self, x, domain_label):
        x = self.conv1(x)
        if self.normalization != 'none':
            if self.normalization == 'dsbn':
                x = self.bn(x, domain_label)
            else:
                x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.normalization != 'none':
            if self.normalization == 'dsbn':
                x = self.bn(x, domain_label)
            else:
                x = self.bn(x)
        x = self.relu(x)

        if self.dropout:
            x = DropBlock(block_size=5, p=0.5)(x)
        return x

class StripConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, direction="horizontal"):
        super(StripConv2d, self).__init__()

        if direction == "horizontal":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=padding)
        elif direction == "vertical":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=padding)
        elif direction == "diagonal1":
            self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
            self.padding = padding
        elif direction == "diagonal2":
            self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
            self.padding = padding
        else:
            raise ValueError("Invalid direction value. Supported values are 'horizontal', 'vertical', 'diagonal1', and 'diagonal2'.")

        self.direction = direction

    def forward(self, x):
        if self.direction in ["horizontal", "vertical"]:
            return self.conv(x)
        elif self.direction == "diagonal1":
            x = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]), mode='reflect')
            return F.conv2d(x, self.weight, stride=1, padding=0, groups=1, dilation=1)
        elif self.direction == "diagonal2":
            x = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]), mode='reflect')
            return F.conv2d(x, self.weight.flip(-1, -2), stride=1, padding=0, groups=1, dilation=1)

class AdaptiveStripUpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(AdaptiveStripUpsamplingBlock, self).__init__()

        padding = (kernel_size // 2, kernel_size // 2)
        self.horizontal_conv = StripConv2d(in_channels, out_channels, kernel_size, padding, direction="horizontal")
        self.vertical_conv = StripConv2d(in_channels, out_channels, kernel_size, padding, direction="vertical")
        self.diagonal_conv1 = StripConv2d(in_channels, out_channels, kernel_size, padding, direction="diagonal1")
        self.diagonal_conv2 = StripConv2d(in_channels, out_channels, kernel_size, padding, direction="diagonal2")

    def forward(self, x):
        # Calculate output size
        h, w = x.size(2) * 2, x.size(3) * 2

        # Apply convolutions
        h_out = self.horizontal_conv(x)
        v_out = self.vertical_conv(x)
        d1_out = self.diagonal_conv1(x)
        d2_out = self.diagonal_conv2(x)

        # Upsample the output tensors
        h_out = F.interpolate(h_out, size=(h, w), mode='bilinear',align_corners=True)
        v_out = F.interpolate(v_out, size=(h, w), mode='bilinear', align_corners=True)
        d1_out = F.interpolate(d1_out, size=(h, w), mode='bilinear', align_corners=True)
        d2_out = F.interpolate(d2_out, size=(h, w), mode='bilinear', align_corners=True)
          # Combine the outputs
        out = h_out + v_out + d1_out + d2_out
        return out


# # UpCatconv(nn.Module) for U-net UP convolution
class UpCatconv(nn.Module):
    def __init__(self, in_feat, out_feat, is_deconv=True, drop_out=False, normalization='none', num_domain = 6):
        super(UpCatconv, self).__init__()
        self.normalization = normalization

        if is_deconv:
            self.conv = conv_block(in_feat, out_feat, drop_out=drop_out, normalization=self.normalization,
                                   num_domain=num_domain)
            # Use ASUB instead of ConvTranspose2d
            self.up = AdaptiveStripUpsamplingBlock(in_feat, out_feat, kernel_size=3)
        else:
            self.conv = conv_block(in_feat + out_feat, out_feat, drop_out=drop_out, normalization=self.normalization,
                                   num_domain=num_domain)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, inputs, down_outputs, domain_label):
        outputs = self.up(down_outputs)
        out = self.conv(torch.cat([inputs, outputs], dim=1), domain_label)

        return out

if __name__ == '__main__':
    import numpy as np
    net_params = {'num_classes':2, 'num_channels':3, 'num_filters':32,
                  'num_filters_cond':16, 'num_domains':6, 'normalization':'dsbn'}
    model = Unet_dsbn_cont(net_params).cuda()
    x = torch.tensor(np.random.random([5, 3, 256, 256]), dtype=torch.float32)
    x = x.cuda()
    pred = model(x,5)
    print(pred[0].shape)