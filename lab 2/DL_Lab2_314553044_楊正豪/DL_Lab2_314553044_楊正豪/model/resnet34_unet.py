import torch
import torch.nn as nn
class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,bias=False),
           nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel,out_channel,3,1,1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        ) 
    def forward(self,x):
        return self.layer(x)
class UpSample(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
    def forward(self,x,feature_map):
        x=self.up(x)
        return torch.cat((feature_map,x),dim=1)
    
class CBAM(nn.Module):
    def __init__(self, in_channel,reduction=16,kernel_size=7):
        super().__init__()
        #channel attention model
        self.max_pool=nn.AdaptiveMaxPool2d(1)
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.mlp=nn.Sequential(
            nn.Conv2d(in_channel,in_channel//reduction,kernel_size=1,bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channel//reduction,in_channel,kernel_size=1,bias=False),
        )
        #spatial
        self.spatial_conv=nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            bias=False
            )
        
        
    
    def forward(self,x):
        #channel attention
        pool_max=self.max_pool(x)
        pool_avg=self.avg_pool(x)

        avg_out=self.mlp(pool_avg)
        max_out=self.mlp(pool_max)

        channel_att=torch.sigmoid(avg_out+max_out)
        x=x*channel_att
        #spatial
        max_pool_spatial,_=torch.max(x,dim=1,keepdim=True)
        avg_pool_spatial=torch.mean(x,dim=1,keepdim=True)
        spatial_input=torch.cat([max_pool_spatial,avg_pool_spatial],dim=1)
        channel_output=self.spatial_conv(spatial_input)
        out=torch.sigmoid(channel_output)
        x=x*out
        return x



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
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        #4 layer
        self.layer1=self._make_layer(64,64,3,1)
        self.layer2=self._make_layer(64,128,4,2)
        self.layer3=self._make_layer(128,256,6,2)
        self.layer4=self._make_layer(256,512,3,2)
        #unet decoder
        self.u1=UpSample(512,256)
        self.c6=Conv_Block(512,32)
        self.cbam1=CBAM(32)

        self.u2=UpSample(32,128)
        self.c7=Conv_Block(256,32)
        self.cbam2=CBAM(32)

        self.u3=UpSample(32,64)
        self.c8=Conv_Block(128,32)
        self.cbam3=CBAM(32)

        self.u4=UpSample(32,32)
        self.c9=Conv_Block(96,32)
        self.cbam4=CBAM(32)

        self.final_up = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.final_conv=nn.Sequential(
            nn.Conv2d(32,32,3,1,1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.out=nn.Conv2d(32,1,1)
    def _make_layer(self,in_channel,out_channel,block,stride):
        if stride != 1 or in_channel!= out_channel:
            shortcut=nn.Sequential(
                nn.Conv2d(in_channel,out_channel,1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channel)
            )
        else:
            shortcut=None
        layers=[]
        layers.append(resnet34_block(in_channel,out_channel,stride,shortcut))
        for _ in range(1,block):
            layers.append(resnet34_block(out_channel,out_channel))
        return nn.Sequential(*layers)
    def forward(self,x):
        #resnet34
        R1=self.pre_layer(x)
        x=self.maxpool(R1)
        R2=self.layer1(x)
        R3=self.layer2(R2)
        R4=self.layer3(R3)
        R5=self.layer4(R4)
        #unet
        Output_one=self.c6(self.u1(R5,R4))
        Output_one=self.cbam1(Output_one)

        Output_two=self.c7(self.u2(Output_one,R3))
        Output_two=self.cbam2(Output_two)

        Output_three=self.c8(self.u3(Output_two,R2))
        Output_three=self.cbam3(Output_three)

        Output_four=self.c9(self.u4(Output_three,R1))
        Output_four=self.cbam4(Output_four)
        #output
        out=self.final_up(Output_four)
        out=self.final_conv(out)
        return self.out(out)    
