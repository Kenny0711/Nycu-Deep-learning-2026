import os
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(256, 256),
    ToTensorV2()
])
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
        else:
            txt_dir=os.path.join(self.root,"annotations","trainval.txt")
        all_fname=[]
        #dog_1 1 1 1 ->dog_1
        with open(txt_dir,"r") as f:
            for line in f:
                line=line.strip()
                part=line.split()
                all_fname.append(part[0])
        #cut train(80%)&&valid(20%)
        trainvalid_part=int(len(all_fname)*0.8)
        if self.mode=="test":
            self.filenames=all_fname
        elif self.mode=="train":
            self.filenames=all_fname[:trainvalid_part]
        elif self.mode=="valid":
            self.filenames=all_fname[trainvalid_part:]
        return self.filenames
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
        mask = self.preprocess_mask(mask)
        if self.transform is not None:
            transformd=self.transform(image=image,mask=mask)
            image=transformd['image']
            mask=transformd['mask']
        picture={"image":image,"mask":mask,"filename":fname}
        return picture
    def preprocess_mask(self,mask):
        mask=mask.astype(np.float32)
        mask[mask==1.0]=1.0
        mask[(mask==2.0)|(mask==3.0)]=0.0
        return mask
#test
if __name__ == "__main__":
    dataset = OxfordPetDataset(
        root="../dataset/oxford-iiit-pet",
        mode="train",
        transform=train_transform
    )
    sample = dataset[0]

    print(type(sample["image"]))
    print(type(sample["mask"]))
    print(sample["image"].shape)
    print(sample["mask"].shape)
    print(sample["filename"])