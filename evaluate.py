import torch
from utils import dice_score 
def evaluate(model, data_loader,DEVICE):
    # implement the evaluation function here
    model.eval()
    num=0
    dice_sum = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            images = batch["image"].float().to(DEVICE, non_blocking=True)
            masks  = batch["mask"].float().to(DEVICE, non_blocking=True)

            logits = model(images)
            probs  = torch.sigmoid(logits)
            preds  = (probs > 0.5).float()                 # 二值化
            dice_sum += dice_score(preds ,masks )
            num +=images.size(0) 
    return dice_sum/num