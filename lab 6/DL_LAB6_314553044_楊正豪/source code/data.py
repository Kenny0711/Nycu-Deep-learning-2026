import os
import json
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import transforms
from torchvision.datasets.folder import default_loader as imgloader
class Object(Dataset):
    def __init__(self, mode='train'):
        super(Object).__init__()
        self.mode = mode
        assert mode in ['train', 'test', 'new_test']
        
        with open('../file/objects.json', 'r') as f:
            self.objects = json.load(f)
            
        with open(f'../file/{mode}.json', 'r') as f:
            self.labels = json.load(f)
            
        if mode == 'train':
            self.images = list(self.labels.keys())
            
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.images)
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            img_filename = self.images[idx]
            img_path = os.path.join('../iclevr', img_filename)
            image = self.transform(imgloader(img_path))
            
            label_name = self.labels[img_filename]
            label = [0] * 24
            for i in label_name: 
                label[self.objects[i]] = 1
            label = torch.tensor(label, dtype=torch.float32)
            return image, label
            
        else:
            label = [0] * 24
            for i in self.labels[idx]: 
                label[self.objects[i]] = 1
            label = torch.tensor(label, dtype=torch.float32)
            return label
