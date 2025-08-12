import torch.nn as nn
import torch.nn.functional as F
import torch
# Implement your UNet model here
# Build one of the main components - DoubleConv - for UNet
class DoubleConv(nn.Module):
    def __init__ (self, in_channels , out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3,stride=1, padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3,stride=1, padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

        )
    def forward(self, x):
        return self.conv(x)
# Build UNet from scrach
class UNet(nn.Module):
    def __init__(self, in_channels=3,out_channels=1, features=[64,128,256,512] ):
        super().__init__()
        self.downs = nn.ModuleList()
        # 在類別（通常是 nn.Module 子類）裡建立一個 
        # 可存放多個子模組（layer）的列表，而且這個列表是 
        # PyTorch 專用的 ModuleList，會自動被框架追蹤參數、放到 GPU、儲存/載入模型。
        self.ups = nn.ModuleList()
        for feature in features:
            self.downs.append(DoubleConv(in_channels,feature))
            in_channels = feature
        self.bottleneck = DoubleConv(features[-1],features[-1]*2)
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))
        self.final_conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels
                                    ,kernel_size=1)
    def forward(self,x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x =F.max_pool2d(x,(2,2))
        x = self.bottleneck(x)
        skip_connections.reverse()
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]
            concat =torch.cat((skip_connection, x), dim=1) # N x C x H x W 把channel cat在一起
            x= self.ups[i+1](concat)
        return self.final_conv(x)