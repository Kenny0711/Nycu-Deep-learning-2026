import torch
import torch.nn as nn
#Conv layer
class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        ) 
    def forward(self,x):
        return self.layer(x)
#DownSample 
class DownSample(nn.Module):
    def __init__(self,channel):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)
#Upsample bilinear 插值表
class UpSample(nn.Module):
    def __init__(self,channel):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
            nn.Conv2d(channel,channel//2,1,1)
        )
    def forward(self,x,feature_map):
        x=self.layer(x)
        return torch.cat((feature_map,x),dim=1)
class Unet(nn.Module):
    def __init__(self,channel):
        super().__init__()
        self.c1=Conv_Block(3,64)
        self.d1=DownSample(64)
        self.c2=Conv_Block(64,128)
        self.d2=DownSample(128)
        self.c3=Conv_Block(128,256)
        self.d3=DownSample(256)
        self.c4=Conv_Block(256,512)
        self.d4=DownSample(512)
        self.c5=Conv_Block(512,1024)
        #bottom
        self.u1=UpSample(1024)
        self.c6=Conv_Block(1024,512)
        self.u2=UpSample(512)
        self.c7=Conv_Block(512,256)
        self.u3=UpSample(256)
        self.c8=Conv_Block(256,128)
        self.u4=UpSample(128)
        self.c9=Conv_Block(128,64)
        self.out=nn.Conv2d(64,3,3,1,1)
        self.sigmoid=nn.Sigmoid()


    def forward(self,x):
        R1=self.c1(x)
        R2=self.c2(self.d1(R1))
        R3=self.c3(self.d2(R2))
        R4=self.c4(self.d3(R3))
        R5=self.c5(self.d4(R4))
        Output_one=self.c6(self.u1(R5,R4))
        Output_two=self.c7(self.u2(Output_one,R3))
        Output_three=self.c8(self.u3(Output_two,R2))
        Output_four=self.c9(self.u4(Output_three,R1))
        return self.sigmoid(self.out(Output_four))
#test
if __name__ == '__main__':
    # 1. 建立我們親手打造的 U-Net 總店 (模型實例化)
    # 輸入通道為 3 (RGB 圖片)
    model = Unet(channel=3)
    
    # 2. 隨機生成一張「假的」圖片張量來當作測試原料
    # PyTorch 的標準形狀是：[Batch_Size, Channels, Height, Width]
    # 我們這裡模擬：一次送進 1 張圖，3 個顏色通道，長寬各為 512
    dummy_input = torch.randn(1, 3, 512, 512)
    
    print("🚀 === 開始測試 U-Net 模型生產線 === 🚀")
    print(f"📥 輸入原料尺寸 (Input Shape): {dummy_input.shape}")
    
    # 3. 按下開關！將假圖片送入模型進行前向傳播 (Forward Pass)
    try:
        output = model(dummy_input)
        print(f"📤 產出結果尺寸 (Output Shape): {output.shape}")
        
        # 4. 驗證最終尺寸是否符合我們期待的 [1, 3, 512, 512]
        if output.shape == (1, 3, 512, 512):
            print("✅ 測試大成功！流水線暢通無阻！")
            print("   模型成功將圖片轉換為 3 個通道的預測遮罩 (狗、背景、邊界)！")
        else:
            print("⚠️ 測試完成，但輸出的尺寸怪怪的喔，請檢查一下。")
            
    except Exception as e:
        print("❌ 糟糕！生產線發生錯誤當機了：")
        print(e)
