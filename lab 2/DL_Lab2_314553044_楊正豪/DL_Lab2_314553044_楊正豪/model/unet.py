import torch
import torch.nn as nn
from utils import center_crop
#捲積層
class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=0,bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=0,bias=True),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)
class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)

    def forward(self, x, feature_map):
        x = self.up(x)
        feature_map = center_crop(feature_map, x)
        return torch.cat((feature_map, x), dim=1)

class unet(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c1 = Conv_Block(channel, 64)
        self.c2 = Conv_Block(64, 128)
        self.c3 = Conv_Block(128, 256)
        self.c4 = Conv_Block(256, 512)
        self.c5 = Conv_Block(512, 1024)
        self.dropout=nn.Dropout(0.5)#paper 3-1 
        self.u1 = UpSample(1024, 512)
        self.c6 = Conv_Block(1024, 512)

        self.u2 = UpSample(512, 256)
        self.c7 = Conv_Block(512, 256)

        self.u3 = UpSample(256, 128)
        self.c8 = Conv_Block(256, 128)

        self.u4 = UpSample(128, 64)
        self.c9 = Conv_Block(128, 64)

        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        r1 = self.c1(x)
        r2 = self.c2(self.pool(r1))
        r3 = self.c3(self.pool(r2))
        r4 = self.c4(self.pool(r3))
        r5 = self.c5(self.pool(r4))
        r5=self.dropout(r5)#paper

        o1 = self.c6(self.u1(r5, r4))
        o2 = self.c7(self.u2(o1, r3))
        o3 = self.c8(self.u3(o2, r2))
        o4 = self.c9(self.u4(o3, r1))

        return self.out(o4)