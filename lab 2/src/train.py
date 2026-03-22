import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from model.unet import* 
from utils import*
from evaluate import evaluate
from oxford_pet import load_dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args():
   parser=argparse.ArgumentParser(description='Train model')
   parser.add_argument('--data_path','-p', type=str, default = '../dataset/oxford-iiit-pet', help='data路徑')
   parser.add_argument('--model','-m',type=str,default='unet',help='unet or resnet')
   parser.add_argument('--epochs','-e',type=int,default=100,help='epoch數量')
   parser.add_argument('--batch_size','-b',type=int,default=16,help='batch size')
   parser.add_argument('--learning_rate','-lr',type=float,default=0.001,help='learing rate')
   parser.add_argument('--load_model_epoch','-lme',type=int,default=0,help='load model epoch')
   return parser.parse_args()

def train(args, model):
    train_loader=load_dataset(
       data_path=args.data_path,
       batch_size=args.batch_size,
       mode="train"
       )
    valid_loader=load_dataset(
       data_path=args.data_path,
       batch_size=args.batch_size,
       mode="valid"
       )
    criterion=bce_dice_loss
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.99)
    #record loss and best dice
    losses=[]
    best_dice = 0.0
    for epoch in range(args.epochs):
      model.train()
      train_loss=0
      for sample in tqdm(train_loader,desc="Train"):
         #forward
         image=sample["image"].float().to(device)
         mask=sample["mask"].float().to(device)
         pred_mask=model(image)
         loss=criterion(pred_mask,mask)
         #backward
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
         train_loss+=loss.item()
        
      train_loss=train_loss/len(train_loader)
      valid_loss,valid_dice=evaluate(model,valid_loader,criterion,device) 
      scheduler.step()   
      print ("Valid time----------")
      print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Valid Loss:{valid_loss:.4f}, Valid Dice:{valid_dice:.4f}")  
      losses.append(train_loss)
      #save model  
      save_dir=f"saved_models/{args.model}"
      os.makedirs(save_dir,exist_ok=True)
      if(epoch+1)%10 ==0:
         torch.save(model.state_dict(), os.path.join(save_dir, f"{args.model}_epoch_{epoch+1}.pth"))
      torch.save(model.state_dict(), os.path.join(save_dir, f"{args.model}_final.pth"))
      #save best dice
      if valid_dice > best_dice:
            print(f"🎉 破紀錄！Valid Dice 從 {best_dice:.4f} 提升到 {valid_dice:.4f}")
            best_dice = valid_dice
            torch.save(model.state_dict(), os.path.join(save_dir, f"{args.model}_best.pth"))
      np.save(os.path.join(save_dir, "train_losses.npy"), losses)

if __name__ == "__main__":
   print("args----------------")
   args=get_args()
   print(args)
   if args.model=="unet":   
      model = unet(channel=3).to(device)
   train(args,model)