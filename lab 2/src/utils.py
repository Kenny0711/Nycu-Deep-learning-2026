import torch
def dice_score(pred_mask,gt_mask):
    pred_mask=pred_mask>0.5
    pred_mask=pred_mask.float()
    gt_mask=gt_mask.float()
    #intersection and union
    intersection=(pred_mask*gt_mask).sum()
    union=pred_mask.sum()+gt_mask.sum()
    dice=2*intersection/(union+1e-8)
    return dice.item()