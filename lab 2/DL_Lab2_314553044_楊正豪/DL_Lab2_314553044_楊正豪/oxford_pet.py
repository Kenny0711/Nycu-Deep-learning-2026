import os
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self,root,mode="train",transform=None):
        assert mode in{"train","valid","test"}
        self.root = root
        self.mode = mode
        self.transform= transform
        #path
        self.image_directory=os.path.join(root,"images")
        self.mask_directory=os.path.join(root,"annotations","trimaps")
        
        self.filenames=self.read_split()
    def read_split(self):
        if self.mode=="test":
            txt_dir=os.path.join(self.root,"annotations","test.txt")
        elif self.mode=="train":
            txt_dir=os.path.join(self.root,"annotations","train.txt")
        elif self.mode=="valid":
            txt_dir=os.path.join(self.root,"annotations","valid.txt")
        all_fname=[]
        #dog_1 1 1 1 ->dog_1
        with open(txt_dir,"r") as f:
            for line in f:
                line=line.strip()
                part=line.split()
                all_fname.append(part[0])
        return all_fname
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, index):
        fname=self.filenames[index]
        #image_path
        image_path=os.path.join(self.image_directory,fname+".jpg")
        mask_path=os.path.join(self.mask_directory,fname+".png")
        #image ->RGB
        image=np.array(Image.open(image_path).convert('RGB'))
        mask=np.array(Image.open(mask_path))
        #紀錄圖片長和寬
        orig_h, orig_w = image.shape[:2]
        mask = self.preprocess_mask(mask)
        #change  to Tensor
        if self.transform is not None:
            transformd=self.transform(image=image,mask=mask)
            image=transformd['image']
            mask=transformd['mask']
        mask = mask.unsqueeze(0).float()
        picture={
            "image":image,
            "mask":mask,
            "filename":fname,
            "orig_h": orig_h, 
            "orig_w": orig_w
            }
        return picture
    def preprocess_mask(self,mask):
        mask=mask.astype(np.float32)
        mask[mask==1.0]=1.0
        mask[(mask==2.0)|(mask==3.0)]=0.0
        return mask
def load_dataset(data_path, batch_size, mode, model_name="unet"):
    mask_size=388
    image_size=572
    if model_name == "unet":
        if mode == 'train':
            transform = A.Compose([
                A.LongestMaxSize(max_size=mask_size),
                A.PadIfNeeded(min_height=mask_size, min_width=mask_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                
                A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_REFLECT_101),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            transform = A.Compose([
                A.LongestMaxSize(max_size=mask_size),
                A.PadIfNeeded(min_height=mask_size, min_width=mask_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_REFLECT_101),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    else: 
        if mode == 'train':
            transform = A.Compose([
                A.Resize(256, 256),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    dataset=OxfordPetDataset(
        root=data_path,
        mode=mode,
        transform=transform
        )
    if mode=='train':
        shuffle=True
    else:
        shuffle=False
    loader=DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=shuffle
        )
    return loader
