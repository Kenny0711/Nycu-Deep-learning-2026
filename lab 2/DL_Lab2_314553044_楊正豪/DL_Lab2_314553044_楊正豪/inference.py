import os
import csv
import torch
import numpy as np
from tqdm import tqdm
import argparse

from model.unet import unet
from model.resnet34_unet import resnet34_unet
from oxford_pet import load_dataset
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--data_path', '-p', type=str, default='../dataset/oxford-iiit-pet')
    parser.add_argument('--model', '-m', type=str, default='unet')
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--model_path', '-mp', type=str, required=True)
    parser.add_argument('--output_path', '-o', type=str, default='submission.csv')
    parser.add_argument('--threshold', '-t', type=float, default=0.5)
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
    pixels = mask.astype(np.uint8).flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def validate_submission_rows(rows):
    issues = []
    row_ids = [row[0] for row in rows]
    if len(row_ids) != len(set(row_ids)):
        issues.append("Duplicated image_id found in submission")
    for image_id, encoded_mask in rows:
        if not image_id:
            issues.append("Found empty image_id")
            break
        if encoded_mask and any(ch not in "0123456789 " for ch in encoded_mask):
            issues.append(f"Invalid RLE format detected for image_id={image_id}")
            break
    return issues


def inference(args, model, test_loader):
    submissions = []

    print("Start inference !!!!!!!!!!!!!!!!")
    with torch.no_grad():
        for sample in tqdm(test_loader, desc="Inference Progress"):
            images = sample["image"].float().to(device)
            filenames = sample["filename"]
            #原本尺寸大小
            orig_hs = sample["orig_h"]
            orig_ws = sample["orig_w"] 
            
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > args.threshold).float()

            for i, fname in enumerate(filenames):
                #紀錄原本圖片長和寬
                h = orig_hs[i].item()
                w = orig_ws[i].item()
                single_pred = preds[i:i+1] 
                if args.model == "unet":
                    scale = 388 / max(h, w)
                    new_h = int(round(h * scale))
                    new_w = int(round(w * scale)) 
                    #切黑邊                 
                    black_top = (388 - new_h) // 2
                    black_left = (388 - new_w) // 2                
                    valid_pred = single_pred[:, :, 
                                            black_top : black_top + new_h,
                                            black_left : black_left + new_w
                                            ]#[B,C,H,W]
                    pred_resized = F.interpolate(valid_pred, size=(h, w), mode='nearest')
                else:
                    pred_resized = F.interpolate(single_pred, size=(h, w), mode='nearest')
                pred_np = pred_resized.squeeze().cpu().numpy().astype(np.uint8)

                encoded_mask = mask_to_rle(pred_np)
                submissions.append((fname, encoded_mask))

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "encoded_mask"])
        writer.writerows(submissions)

    print("=" * 60)
    print("Inference complete")
    print(f"Model checkpoint: {args.model_path}")
    print(f"Total test images: {len(submissions)}")
    print(f"Submission saved to: {args.output_path}")
    issues = validate_submission_rows(submissions)
    if issues:
        print("Kaggle format check: FAILED")
        for issue in issues:
            print(f" - {issue}")
    else:
        print("Kaggle format check: PASSED")
    print("=" * 60)


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