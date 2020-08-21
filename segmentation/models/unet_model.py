# Taken from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16) # was 64
        self.down1 = Down(16, 32) # was 64, 128
        self.down2 = Down(32, 64) # was 128, 256
        self.down3 = Down(64, 128) # was 256,  512
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor) # was 512, 1024
        self.up1 = Up(256, 128 // factor, bilinear) # was 1024, 512
        self.up2 = Up(128, 64 // factor, bilinear) # was 512, 256
        self.up3 = Up(64, 32 // factor, bilinear) # was 256, 128
        self.up4 = Up(32, 16, bilinear) # was 128, 64
        self.outc = OutConv(16, n_classes) # was 64

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits