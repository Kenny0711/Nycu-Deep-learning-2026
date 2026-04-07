import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
#
from torch.utils.tensorboard import SummaryWriter

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        #
        self.args=args
        #
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers()
        self.prepare_training()
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self,train_loader):
        self.model.train()
        pbar=tqdm(train_loader)
        total_loss=0
        for _,x in enumerate(pbar):
            x=x.to(self.args.device)
            self.optim.zero_grad()
            ratio = np.random.rand()
            logits, z_indices = self.model(x, ratio)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), 
                z_indices.reshape(-1).long()
            )
            loss.backward()
            self.optim.step()
            total_loss+=loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss/len(train_loader)

    def eval_one_epoch(self,val_loader):
        self.model.eval()
        pbar=tqdm(val_loader)
        total_loss=0
        with torch.no_grad():
            for _,x in enumerate(pbar):
                x=x.to(self.args.device)
                #[B,256,1025]、[B,256]
                ratio = np.random.rand()
                logits, z_indices = self.model(x, ratio)
                loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), 
                z_indices.reshape(-1).long()
            )  
                total_loss+=loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss/len(val_loader)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.transformer.parameters(),
            lr=self.args.learning_rate,
            )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=self.args.epochs)
        return optimizer,scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="lab3_dataset/lab3_dataset/train", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="lab3_dataset/lab3_dataset/val", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers','-n', type=int, default=0, help='Number of worker')
    parser.add_argument('--batch-size','-b', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=10, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=1, help='Number of epochs to train.')
    parser.add_argument('--learning-rate','-lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight Decay.')
    parser.add_argument('--warmup-steps','-w', type=int, default=1, help='Warmup steps.')
    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5: 
#
    writer = SummaryWriter('transformer_checkpoints/logs/')
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        train_loss=train_transformer.train_one_epoch(train_loader)
        valid_loss=train_transformer.eval_one_epoch(val_loader)
        writer.add_scalar('train_loss',train_loss,epoch)
        writer.add_scalar('valid_loss',valid_loss,epoch)
        writer.add_scalar('Learning Rate', train_transformer.scheduler.get_last_lr()[-1], epoch)
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f} | LR: {train_transformer.scheduler.get_last_lr()[0]:5f}")
        train_transformer.scheduler.step()
        if epoch%args.save_per_epoch==0:
            torch.save(train_transformer.model.transformer.state_dict(), f"transformer_checkpoints/ckpt_{epoch}.pt")
            print(f"Model saved at epoch {epoch}")