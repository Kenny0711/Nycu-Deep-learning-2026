import os
import sys
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from diffusers import DDPMScheduler
from torchvision.transforms import Normalize
import numpy as np

from unet import Unet
from data import Object

sys.path.append(os.path.abspath("../file"))
from evaluator import evaluation_model
def generate_denoising_visualization(args, model, noise_scheduler, objects_map, output_dir, device):
    prompt = ["red sphere", "cyan cylinder", "cyan cube"]

    labels = torch.zeros(1, 24, device=device)
    for label_text in prompt:
        if label_text in objects_map:
            labels[0, objects_map[label_text]] = 1

    image = torch.randn((1, 3, 64, 64), device=device)
    
    denoise_process_images = [image.clone()]

    num_inference_steps = len(noise_scheduler.timesteps)
    save_indices = np.linspace(1, num_inference_steps - 1, num=9, dtype=int) 

    with torch.no_grad():
        for i, t in enumerate(tqdm(noise_scheduler.timesteps, desc="Generating denoising process")):
            
            latent_model_input = torch.cat([image] * 2)
            uncond_labels = torch.zeros_like(labels)
            combined_labels = torch.cat([labels, uncond_labels], dim=0)

            t_val = t.item() if isinstance(t, torch.Tensor) else t
            t_for_model = torch.full((latent_model_input.shape[0],), t_val, device=device, dtype=torch.long)
            
            noise_pred = model(latent_model_input, t_for_model, combined_labels)
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            
            guided_noise = noise_pred_uncond + args.cfg_scale * (noise_pred_cond - noise_pred_uncond)

            image = noise_scheduler.step(guided_noise, t, image).prev_sample

            if i in save_indices:
                denoise_process_images.append(image.clone())

    denoise_process_images = [(img.clamp(-1, 1) + 1) / 2 for img in denoise_process_images]
    grid = make_grid(torch.cat(denoise_process_images), nrow=len(denoise_process_images), padding=2)
    
    save_path = os.path.join(output_dir, "denoising_process.png")
    save_image(grid, save_path)
    print(f"Save_path:{save_path}")


def inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Unet().to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=500, beta_schedule="squaredcos_cap_v2")
    noise_scheduler.set_timesteps(500)
    
    test_modes = ['test', 'new_test']
    evaluator = evaluation_model()
    
    for mode in test_modes:
        print(f"\n Mode: {mode}")
        dataset = Object(mode=mode)
        labels = torch.stack([dataset[i] for i in range(len(dataset))]).to(device)
        
        save_dir = os.path.join(args.output_dir, mode)
        os.makedirs(save_dir, exist_ok=True)
        
        with torch.no_grad():
            current_images = torch.randn(len(labels), 3, 64, 64).to(device)
            uncond_labels = torch.zeros_like(labels).to(device)
            
            for t in tqdm(noise_scheduler.timesteps, desc="Denoising"):
                latent_model_input = torch.cat([current_images, current_images], dim=0)
                combined_labels = torch.cat([labels, uncond_labels], dim=0)
                
                t_val = t.item() if isinstance(t, torch.Tensor) else t
                t_for_model = torch.full((latent_model_input.shape[0],), t_val, device=device, dtype=torch.long)
                
                noise_pred = model(latent_model_input, t_for_model, combined_labels)
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
                
                noise_pred = noise_pred_uncond + args.cfg_scale * (noise_pred_cond - noise_pred_uncond)
                current_images = noise_scheduler.step(noise_pred, t, current_images).prev_sample
        
        final_images = (current_images / 2 + 0.5).clamp(0, 1)
        eval_images = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(final_images)
        
        acc = evaluator.eval(eval_images, labels)
        print(f"{mode} Accuracy: {acc:.4f}")

        grid = make_grid(final_images, nrow=8)
        save_image(grid, f"{args.output_dir}/{mode}_grid.png")
        
        for i in range(len(final_images)):
            save_image(final_images[i], f"{save_dir}/{i+1}.png")


    
    with open("../file/objects.json", "r") as f:
        objects_map = json.load(f)
    

    generate_denoising_visualization(args, model, noise_scheduler, objects_map, args.output_dir, device)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints-3/best_model.pth")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--cfg_scale", type=float, default=10.0)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    inference(args)