import torch
import torch.nn as nn
from model.unet import Conv_Block,UpSample
class resnet34_block(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1,shortcut=None):
        super().__init__()
        # 2 3x3
        self.conv1=nn.Conv2d(in_channel,out_channel,3,stride,1,bias=False)
        self.b1=nn.BatchNorm2d(out_channel)
        self.r1=nn.ReLU()

        self.conv2=nn.Conv2d(out_channel,out_channel,3,1,1,bias=False)
        self.b2=nn.BatchNorm2d(out_channel)
        self.shortcut=shortcut
        self.stride=stride
    def forward(self,x):
        #捷徑
        if self.shortcut is not None:
            identify=self.shortcut(x)
        else:
            identify=x
        out=self.conv1(x)
        out=self.b1(out)
        out=self.r1(out)
        out=self.conv2(out)
        out=self.b2(out)
        out+=identify
        out=self.r1(out)
        return out
class resnet34_unet(nn.Module):
    def __init__(self,channel):
        super().__init__()
        #前置層 7x7
        self.pre_layer=nn.Sequential(
            nn.Conv2d(channel,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool=nn.Sequential(
            nn.Conv2d(64,64,3,2,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        #4 layer
        self.layer1=self._make_layer(64,64,3,1)
        self.layer2=self._make_layer(64,128,4,2)
        self.layer3=self._make_layer(128,256,6,2)
        self.layer4=self._make_layer(256,512,3,2)
        #unet decoder
        self.u1=UpSample(512,256)
        self.c6=Conv_Block(512,256)
        self.u2=UpSample(256,128)
        self.c7=Conv_Block(256,128)
        self.u3=UpSample(128,64)
        self.c8=Conv_Block(128,64)
        self.u4=UpSample(64,64)
        self.c9=Conv_Block(128,64)
        self.final = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.out=nn.Conv2d(64,1,1)
    def _make_layer(self,in_channel,out_channel,block,stride,is_shortcut=True):
        shortcut=None
        if is_shortcut:
            shortcut=nn.Sequential(
                nn.Conv2d(in_channel,out_channel,1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channel)
            )
        layers=[]
        layers.append(resnet34_block(in_channel,out_channel,stride,shortcut))
        for _ in range(1,block):
            layers.append(resnet34_block(out_channel,out_channel))
        return nn.Sequential(*layers)
    def forward(self,x):
        #resnet34
        R1=self.pre_layer(x)
        x=self.pool(R1)
        R2=self.layer1(x)
        R3=self.layer2(R2)
        R4=self.layer3(R3)
        R5=self.layer4(R4)
        #unet
        Output_one=self.c6(self.u1(R5,R4))
        Output_two=self.c7(self.u2(Output_one,R3))
        Output_three=self.c8(self.u3(Output_two,R2))
        Output_four=self.c9(self.u4(Output_three,R1))
        #output
        out=self.final(Output_four)
        return self.out(out)    
