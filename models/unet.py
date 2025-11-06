import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x_before_pool = self.conv(x)
        x_after_pool = self.pool(x_before_pool)
        return x_before_pool, x_after_pool

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, x_skip):
        x = self.up(x)
        x = torch.cat([x, x_skip], dim=1)
        x = self.conv(x)
        return x

class ClimateUNet(nn.Module):
    def __init__(self, in_channels=13, num_classes=5, base_channels=64):
        super().__init__()
        
        self.down1 = DownBlock(in_channels, base_channels)
        self.down2 = DownBlock(base_channels, base_channels * 2)
        self.down3 = DownBlock(base_channels * 2, base_channels * 4)
        self.down4 = DownBlock(base_channels * 4, base_channels * 8)
        
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)
        
        self.up4 = UpBlock(base_channels * 16, base_channels * 8)
        self.up3 = UpBlock(base_channels * 8, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels)
        
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        
    def forward(self, x):
        x1, x = self.down1(x)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)
        
        x = self.bottleneck(x)
        
        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        
        x = self.final_conv(x)
        return x