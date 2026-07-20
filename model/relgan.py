# Copyright (C) 2019 Willy Po-Wei Wu & Elvis Yu-Jing Lin <maya6282@gmail.com, elvisyjlin@gmail.com>
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.


import random
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from net.switchable_norm import SwitchNorm2d


def tile_like(x, target):  # tile_size = 256 or 4
    x = x.view(x.size(0), x.size(1), 1, 1)
    x = x.repeat(1, 1, target.size(2), target.size(3))
    return x

def count_trainable_parameters(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])



class ResidualBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size):
        super(ResidualBlock, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size=kernel_size, padding=1, stride=1), 
            SwitchNorm2d(n_out, momentum=0.9),
            #nn.InstanceNorm2d(n_out, affine=True, track_running_stats=True), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(n_out, n_out, kernel_size=kernel_size, padding=1, stride=1), 
            SwitchNorm2d(n_out, momentum=0.9),
            #nn.InstanceNorm2d(n_out, affine=True, track_running_stats=True),  
        )
    def forward(self, x):
        return x + self.f(x)


class G(nn.Module):
    def __init__(self, n_c=3, n_z=5, n_repeat=6):
        super(G, self).__init__()
        self.n_z = n_z
        
        self.conv_in = nn.Sequential(
            nn.Conv2d(n_c + n_z, 64, kernel_size=7, padding=3, stride=1),
            SwitchNorm2d(64, momentum=0.9), 
            #nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True), 
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2), 
            SwitchNorm2d(128, momentum=0.9), 
            #nn.InstanceNorm2d(128, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True), 
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=2), 
            SwitchNorm2d(256, momentum=0.9),
            #nn.InstanceNorm2d(256, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True), 
        )
        resb_layers = [ResidualBlock(256, 256, 3) for _ in range(n_repeat)]
        self.resb = nn.Sequential(*resb_layers)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, padding=1, stride=2), 
            SwitchNorm2d(128, momentum=0.9),
            #nn.InstanceNorm2d(128, affine=True, track_running_stats=True), 
            nn.ReLU(inplace=True), 
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2), 
            SwitchNorm2d(64, momentum=0.9),
            #nn.InstanceNorm2d(64, affine=True, track_running_stats=True), 
            nn.ReLU(inplace=True), 
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(64, n_c, kernel_size=7, padding=3, stride=1), 
            nn.Tanh(), 
        )
    def forward(self, img, z):
        tiled_z = tile_like(z, img)
        x = torch.cat([img, tiled_z], dim=1)
        h = self.conv_in(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.resb(h)
        h = self.up2(h)
        h = self.up1(h)
        y = self.conv_out(h)
        return y
    
    def mid_features(self, img, z):
        tiled_z = tile_like(z, img)
        x = torch.cat([img, tiled_z], dim=1)
        h = self.conv_in(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.resb[0](h)
        h = self.resb[1](h)
        y = self.resb[2](h)
        return y
    

class ResidualBlockg(nn.Module):
    def __init__(self, n_in, n_out, kernel_size):
        super(ResidualBlockg, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size=kernel_size, padding=1, stride=1,bias=False), 
            #SwitchNorm2d(n_out, momentum=0.9),
            nn.InstanceNorm2d(n_out, affine=True, track_running_stats=True), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(n_out, n_out, kernel_size=kernel_size, padding=1, stride=1,bias=False), 
            #SwitchNorm2d(n_out, momentum=0.9),
            nn.InstanceNorm2d(n_out, affine=True, track_running_stats=True),  
        )
    def forward(self, x):
        return x + self.f(x)



class Gg(nn.Module):
    def __init__(self, n_c=3, n_z=5, n_repeat=6):
        super(Gg, self).__init__()
        self.n_z = n_z
        
        self.conv_in = nn.Sequential(
            nn.Conv2d(n_c + n_z, 64, kernel_size=7, padding=3, stride=1,bias=False),
            #SwitchNorm2d(64, momentum=0.9), 
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True), 
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2,bias=False), 
            #SwitchNorm2d(128, momentum=0.9), 
            nn.InstanceNorm2d(128, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True), 
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=2,bias=False), 
            #SwitchNorm2d(256, momentum=0.9),
            nn.InstanceNorm2d(256, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True), 
        )
        resb_layers = [ResidualBlockg(256, 256, 3) for _ in range(n_repeat)]
        self.resb = nn.Sequential(*resb_layers)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, padding=1, stride=2,bias=False), 
            #SwitchNorm2d(128, momentum=0.9),
            nn.InstanceNorm2d(128, affine=True, track_running_stats=True), 
            nn.ReLU(inplace=True), 
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2,bias=False), 
            #SwitchNorm2d(64, momentum=0.9),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True), 
            nn.ReLU(inplace=True), 
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(64, n_c, kernel_size=7, padding=3, stride=1,bias=False), 
            nn.Tanh(), 
        )
    def forward(self, img, z):
        tiled_z = tile_like(z, img)
        x = torch.cat([img, tiled_z], dim=1)
        h = self.conv_in(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.resb(h)
        h = self.up2(h)
        h = self.up1(h)
        y = self.conv_out(h)
        return y



class D(nn.Module):
    def __init__(self, n_c, n_z, n_filters=[64, 128, 256, 512, 1024, 2048]):
        print('Building discriminator...')
        super(D, self).__init__()
        layers = []
        n_in = n_c
        for n_f in n_filters:
            layers += [nn.Conv2d(n_in, n_f, kernel_size=4, padding=1, stride=2)]
            layers += [nn.LeakyReLU(negative_slope=0.01, inplace=True)]
            n_in = n_f
        self.convs = nn.Sequential(*layers)
        self.conv_adv = nn.Conv2d(n_in, 1, kernel_size=1, padding=0, stride=1)
        self.conv_int = nn.Conv2d(n_in, 16, kernel_size=1, padding=0, stride=1)
        n_in_c = n_filters[-1] * 2 + n_z
        self.convs_cls = nn.Sequential(
            nn.Conv2d(n_in_c, 2048, kernel_size=1, padding=0, stride=1), 
            nn.LeakyReLU(negative_slope=0.01, inplace=True), 
            nn.Conv2d(2048, 1, kernel_size=1, padding=0, stride=1), 
        )
    def forward(self, img_a, img_b=None, z=None, critic=False):
        if not critic:
            assert img_a is not None and img_b is not None and z is not None
            h_a = self.convs(img_a)
            h_b = self.convs(img_b)
            y_1 = self.conv_adv(h_b)
            tiled_z = tile_like(z, h_a)
            h = torch.cat([h_a, h_b, tiled_z], dim=1)
            y_2 = self.convs_cls(h)
            return y_1, y_2
        else:
            assert img_a is not None
            h = self.convs(img_a)
            h = self.conv_int(h)
            y = h.view(h.size(0), -1).mean(1, keepdim=True)  # Global average pooling
            return y