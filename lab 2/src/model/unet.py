import torch
import torch.nn as nn
#Conv layer
class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,bias=False),
#            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel,out_channel,3,1,1,bias=False),
#            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        ) 
    def forward(self,x):
        return self.layer(x)
#DownSample 
# class DownSample(nn.Module):
#     def __init__(self,channel):
#         super().__init__()
#         self.layer=nn.Sequential(
#             nn.Conv2d(channel,channel,3,2,1,bias=False),
#             nn.BatchNorm2d(channel),
#             nn.ReLU()
#         )
#     def forward(self,x):
#         return self.layer(x)
#Upsample bilinear 插值表
class UpSample(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
    def forward(self,x,feature_map):
        x=self.up(x)
        return torch.cat((feature_map,x),dim=1)
    
class unet(nn.Module):
    def __init__(self,channel):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #down
        self.c1=Conv_Block(channel,64)
        self.c2=Conv_Block(64,128)
        self.c3=Conv_Block(128,256)
        self.c4=Conv_Block(256,512)
        self.c5=Conv_Block(512,1024)
        #Up
        self.u1=UpSample(1024,512)
        self.c6=Conv_Block(1024,512)
        self.u2=UpSample(512,256)
        self.c7=Conv_Block(512,256)
        self.u3=UpSample(256,128)
        self.c8=Conv_Block(256,128)
        self.u4=UpSample(128,64)
        self.c9=Conv_Block(128,64)
        self.out=nn.Conv2d(64,1,1)


    def forward(self,x):
        R1=self.c1(x)
        R2=self.c2(self.pool(R1))
        R3=self.c3(self.pool(R2))
        R4=self.c4(self.pool(R3))
        R5=self.c5(self.pool(R4))
        Output_one=self.c6(self.u1(R5,R4))
        Output_two=self.c7(self.u2(Output_one,R3))
        Output_three=self.c8(self.u3(Output_two,R2))
        Output_four=self.c9(self.u4(Output_three,R1))
        return self.out(Output_four)
