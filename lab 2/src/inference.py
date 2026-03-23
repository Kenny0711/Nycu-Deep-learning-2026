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
    parser = argparse.ArgumentParser(description='Inference for Lab2')
    parser.add_argument('--data_path', '-p', type=str, default='../dataset/oxford-iiit-pet', help='資料集路徑')
    parser.add_argument('--model', '-m', type=str, default='unet', choices=['unet', 'resnet34_unet'], help='使用的模型架構')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='Batch size')
    parser.add_argument('--model_path', '-mp', type=str, required=True, help='訓練好的權重檔路徑 (.pth)')
    parser.add_argument('--output_path', '-o', type=str, default='submission.csv', help='輸出的 Kaggle CSV 檔名')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='sigmoid threshold')
    return parser.parse_args()


def build_model(model_name):
    if model_name == "unet":
        return unet(channel=3)
    elif model_name == "resnet34_unet":
        return resnet34_unet(channel=3)
    else:
        raise ValueError("不支援的模型架構！")


def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def mask_to_rle(mask):
    """
    將二元遮罩轉成 Kaggle 常用 RLE
    使用 column-major / Fortran order
    """
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

    print(f"🚀 開始進行推論，使用裝置: {device}")
    with torch.no_grad():
        for sample in tqdm(test_loader, desc="Inference Progress"):
            images = sample["image"].float().to(device)
            filenames = sample["filename"]
            orig_hs = sample["orig_h"] # 🌟 取得這批圖片的原始高度
            orig_ws = sample["orig_w"] # 🌟 取得這批圖片的原始寬度
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > args.threshold).float()

            for i, fname in enumerate(filenames):
                h = orig_hs[i].item()
                w = orig_ws[i].item()

                # 取出單張預測圖 [1, 1, 256, 256]
                single_pred = preds[i:i+1] 

                # 🌟 關鍵修復：使用 nearest (最近鄰) 插值法放大回 (h, w)
                pred_resized = F.interpolate(single_pred, size=(h, w), mode='nearest')
                
                # 轉成 numpy 並拿掉多餘的維度
                pred_np = pred_resized.squeeze().cpu().numpy().astype(np.uint8)

                # 編碼並儲存
                encoded_mask = mask_to_rle(pred_np)
                submissions.append((fname, encoded_mask))

    issues = validate_submission_rows(submissions)

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