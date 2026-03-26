import torch
import torch.nn as nn
def dice_score(pred_mask, gt_mask,threshold=0.5):
    pred_mask = torch.sigmoid(pred_mask)          
    pred_mask = (pred_mask > threshold).float()    
    #test1
    pred_mask = pred_mask.view(pred_mask.size(0), -1)
    gt_mask = gt_mask.view(gt_mask.size(0), -1)
    #
    gt_mask = gt_mask.float()
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    dice = 2 * intersection / (union + 1e-8)
    return dice.mean().item()

def dice_loss(pred_mask, gt_mask):
    pred_mask = torch.sigmoid(pred_mask)          
    gt_mask = gt_mask.float()
    #test1
    pred_mask = pred_mask.view(pred_mask.size(0), -1)
    gt_mask = gt_mask.view(gt_mask.size(0), -1)
    #
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    dice = 2 * intersection  / (union + 1e-8)
    return 1 - dice.mean()
def bce_dice_loss(pred_mask, gt_mask):
    criterion_bce = nn.BCEWithLogitsLoss()
    bce = criterion_bce(pred_mask, gt_mask)
    dloss = dice_loss(pred_mask, gt_mask)
    return bce + dloss