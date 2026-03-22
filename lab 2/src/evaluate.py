import torch
from tqdm import tqdm 
from utils import *
def evaluate(model, valid_loader, criterion, device):
    model.eval()
    valid_loss = 0
    valid_dice=0
    with torch.no_grad():
        for sample in tqdm(valid_loader,desc="Valid:"):
            image = sample["image"].float().to(device)
            mask = sample["mask"].float().to(device)

            pred_mask = model(image)
            loss = criterion(pred_mask, mask)
            valid_loss += loss.item()
            valid_dice+=dice_score(pred_mask,mask)

    valid_loss = valid_loss / len(valid_loader)
    valid_dice = valid_dice / len(valid_loader)
    return valid_loss,valid_dice