import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from model.unet import* 
from model.resnet34_unet import*
from utils import*
from evaluate import evaluate
from oxford_pet import load_dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#hyerparameters
def get_args():
   parser=argparse.ArgumentParser(description='Train model')
   parser.add_argument('--data_path','-p', type=str, default = '../dataset/oxford-iiit-pet')
   parser.add_argument('--model','-m',type=str,default='unet')
   parser.add_argument('--epochs','-e',type=int,default=100)
   parser.add_argument('--batch_size','-b',type=int,default=16)
   parser.add_argument('--learning_rate','-lr',type=float,default=0.001)
   parser.add_argument('--threshold', '-t', type=float, default=0.5)
   parser.add_argument('--load_model', '-load', type=str, default=None, help='載入已訓練權重')
   return parser.parse_args()

def train(args, model):
   train_loader=load_dataset(
       data_path=args.data_path,
       batch_size=args.batch_size,
       mode="train",
       model_name=args.model
       )
   valid_loader=load_dataset(
       data_path=args.data_path,
       batch_size=args.batch_size,
       mode="valid",
       model_name=args.model
       )
   optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4)
   scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.99)
   #record loss and best dice
   train_losses = []
   valid_losses = []
   valid_dices = []
   best_dice = 0.0
   criterion=bce_dice_loss
   for epoch in range(args.epochs):
      model.train()
      train_loss=0
      for sample in tqdm(train_loader,desc="Train"):
         #forward
         image=sample["image"].float().to(device)
         mask=sample["mask"].float().to(device)

         pred_mask=model(image)
         if args.model == "unet":
                mask = center_crop(mask, pred_mask)
         loss=criterion(pred_mask, mask, model_name=args.model)
         #backward
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
         train_loss+=loss.item()
        
      train_loss=train_loss/len(train_loader)
      valid_loss,valid_dice=evaluate(model,valid_loader,criterion,args.threshold,device,args.model) 
      scheduler.step()   
      print ("Valid time----------")
      print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Valid Loss:{valid_loss:.4f}, Valid Dice:{valid_dice:.4f}")
      train_losses.append(train_loss)
      valid_losses.append(valid_loss)
      valid_dices.append(valid_dice)  
      #save model  
      save_dir=f"saved_models/{args.model}"
      os.makedirs(save_dir,exist_ok=True)
      if(epoch+1)%10 ==0:
         torch.save(model.state_dict(), os.path.join(save_dir, f"{args.model}_epoch_{epoch+1}.pth"))
      torch.save(model.state_dict(), os.path.join(save_dir, f"{args.model}_final.pth"))
      #save best dice
      if valid_dice > best_dice:
            print(f"Best Valid Dice!   {best_dice:.4f} ------> {valid_dice:.4f}")
            best_dice = valid_dice
            torch.save(model.state_dict(), os.path.join(save_dir, f"{args.model}_best.pth"))

if __name__ == "__main__":
   print("args----------------")
   args=get_args()
   print(args)
   if args.model=="unet":   
      model = unet(channel=3).to(device)
   if args.model=="resnet34_unet":
      model=resnet34_unet(channel=3).to(device)
   train(args,model)