import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb

from data import Object 
from unet import Unet

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("-" * 30)
    print("Training Setting:")
    for key, value in vars(args).items():
        print(f"{key:>15}: {value}")
    print("-" * 30)

    os.makedirs(args.save_dir, exist_ok=True)

    print("Load dataset")
    train_dataset = Object(mode='train')
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = Unet().to(device)
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.timesteps, beta_schedule="squaredcos_cap_v2") 
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    total_training_steps = len(train_loader) * args.num_epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=total_training_steps
    )

    global_step = 0
    best_loss = float('inf')  

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{args.num_epochs:03d}")
        
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).float()
            bs = images.shape[0]

            # CFG Dropout
            if args.cfg_drop_rate > 0:
                drop_mask = torch.rand(bs, 1, device=device) < args.cfg_drop_rate
                labels = torch.where(drop_mask, torch.zeros_like(labels), labels)

            noise = torch.randn_like(images).to(device)

            t = torch.randint(0, args.timesteps, (bs,), device=device).long()

            noisy_images = noise_scheduler.add_noise(images, noise, t)

            pred_noise = model(noisy_images, t, labels)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step() 

            epoch_loss += loss.item()
            global_step += 1
            
            current_lr = lr_scheduler.get_last_lr()[0]
            wandb.log({
                "Train/Step_Loss": loss.item(),
                "Train/Learning_Rate": current_lr
            }, step=global_step)
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{current_lr:.6f}"})

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} finish | Loss: {avg_loss:.4f}")
        wandb.log({"Train/Epoch_Avg_Loss": avg_loss, "Epoch": epoch + 1}, step=global_step)
        
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.num_epochs:
            save_path = os.path.join(args.save_dir, f"ddpm_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"save path: {save_path}")
            
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 6 DDPM Training")
    
    # training setting
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    
    # Diffusion setting
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--cfg_drop_rate", type=float, default=0.1)
    
    # save
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="checkpoints-3")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--wandb_run_name", type=str, default="Lab6_Identity_Embedding")
    
    args = parser.parse_args()
    wandb.init(project="DLP-Lab6", name=args.wandb_run_name, config=vars(args), save_code=True)
    train(args)