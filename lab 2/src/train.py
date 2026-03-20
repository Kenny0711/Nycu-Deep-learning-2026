import torch
import json
from torch  import nn,optim
from torch.utils.data import DataLoader
from oxford_pet import *
from model.unet import *
from tqdm import tqdm
def train(config,data_path,model):
    config["data_path"]=data_path
    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'
    print(f"現在使用的設備:{device}")

    model_path=os.path.join('./saved_models/train')
    train_loader=load_dataset(config,mode="train")
    valid_loader=load_dataset(config,mode="valid")
    criterion=nn.BCELoss()
    optimzer=torch.optim.adam(model.paramter(),lr=config["learning_rate"])
    scheduler=

if __name__=='__main__':
   