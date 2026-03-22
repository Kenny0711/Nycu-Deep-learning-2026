import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

from model.unet import unet
from model.resnet34_unet import resnet34_unet
from oxford_pet import load_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--data_path', '-p', type=str, default='../dataset/oxford-iiit-pet')
    parser.add_argument('--model', '-m', type=str, default='unet')
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--model_path', '-mp', type=str, required=True)
    parser.add_argument('--output_path', '-o', type=str, default='submission.csv')
    return parser.parse_args()


def build_model(model_name):
    if model_name == "unet":
        return unet(channel=3)
    elif model_name == "resnet34_unet":
        return resnet34_unet(channel=3)


def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def mask_to_rle(mask):
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
def inference(args, model, test_loader):
    ids = []
    rles = []
    with torch.no_grad():
        for sample in tqdm(test_loader, desc="Inference"):
            images = sample["image"].float().to(device)
            filenames = sample["filename"]
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            preds = preds.squeeze(1).cpu().numpy().astype(np.uint8)
            for fname, pred in zip(filenames, preds):
                ids.append(fname)
                rles.append(mask_to_rle(pred))
    df = pd.DataFrame({
        "id": ids,
        "rle_mask": rles
    })
    df.to_csv(args.output_path, index=False)
    print(f"✅ 下載到{args.output_path}")
if __name__ == "__main__":
    args = get_args()
    model = build_model(args.model)
    model = load_model(model, args.model_path)
    test_loader = load_dataset(
        data_path=args.data_path,
        batch_size=args.batch_size,
        mode="test"
    )
    inference(args, model, test_loader)