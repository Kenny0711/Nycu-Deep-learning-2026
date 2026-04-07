import torch
import torch.nn as nn

def dice_score(pred_mask, gt_mask, threshold=0.5):
    pred_mask = torch.sigmoid(pred_mask)          
    pred_mask = (pred_mask > threshold).float()    
    gt_mask = gt_mask.float()
    
    pred_mask = pred_mask.view(pred_mask.size(0), -1)
    gt_mask = gt_mask.view(gt_mask.size(0), -1)
    
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    dice = 2 * intersection / (union + 1e-8)
    return dice.mean().item()

def dice_loss(pred_mask, gt_mask):
    pred_mask = torch.sigmoid(pred_mask)          
    gt_mask = gt_mask.float()
    
    pred_mask = pred_mask.view(pred_mask.size(0), -1)
    gt_mask = gt_mask.view(gt_mask.size(0), -1)
    
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    dice = 2 * intersection  / (union + 1e-8)
    return 1 - dice.mean()

def bce_dice_loss(pred_mask, gt_mask, model_name="unet"):
    criterion_bce = nn.BCEWithLogitsLoss()
    bce = criterion_bce(pred_mask, gt_mask)
    dloss = dice_loss(pred_mask, gt_mask)

    if model_name == "unet":
        return 0.3 * bce + 0.7 * dloss
    else:
        return bce + dloss
#尺寸不一樣的結合
def center_crop(feature_map, target_tensor):
    _, _, h, w = target_tensor.shape
    _, _, H, W = feature_map.shape
    start_h = (H - h) // 2
    start_w = (W - w) // 2
    return feature_map[:, :, start_h:start_h+h, start_w:start_w+w]