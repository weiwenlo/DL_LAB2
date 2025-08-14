import argparse
import torch
from torch.utils.data import DataLoader
from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import ResNetU,BasicBlock
from tqdm import tqdm
from utils import dice_score 
import matplotlib.pyplot as plt
import os
def inference(args):
    data_path= args.data_path
    batch_size = args.batch_size
    model_para_path = args.model_para_path
    save_dir = args.save_img_dir

    test_dataset = load_dataset(data_path,"test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,              # Windows 也 OK；如果記憶體吃緊可降到 2
        pin_memory=True,
        persistent_workers=True     # 需要 num_workers > 0
    )
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = UNet(in_channels=3, out_channels=1)
    model = ResNetU(BasicBlock, layers=[3, 4, 6, 3], num_classes=1)
    model = model.to(DEVICE)

    state = torch.load(model_para_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    pbar = tqdm(test_loader, desc=f"testing ", ncols=100)

    num=0
    dice_sum = 0
    first_batch_images = None
    first_batch_gt = None
    first_batch_pred = None
    for batch_idx, batch in enumerate(pbar):

        images = batch["image"].float().to(DEVICE, non_blocking=True)
        masks  = batch["mask"].float().to(DEVICE, non_blocking=True)
        logits = model(images)                             # (N,1,H,W)
        probs  = torch.sigmoid(logits)
        preds  = (probs > 0.5).float()

        dice_sum += dice_score(preds ,masks )

        num +=images.size(0)
        if batch_idx == 0:
            first_batch_images = images
            first_batch_gt = masks
            first_batch_pred = preds
    print(f"[Inference] Arch= | Avg Dice on TEST: {dice_sum/num:.6f} (N={num})")
    if first_batch_images is not None:
        visualize_triplets(
            images=first_batch_images,
            gt_masks=first_batch_gt,
            pred_masks=first_batch_pred,
            save_dir=save_dir,
            max_show=4
        )
def visualize_triplets(images, gt_masks, pred_masks, save_dir=None, max_show=4):
    """
    images:      Tensor (N, 3, H, W)
    gt_masks:    Tensor (N, 1, H, W)
    pred_masks:  Tensor (N, 1, H, W)  # 已經二值化
    """
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    images = images.cpu()
    gt_masks = gt_masks.cpu()
    pred_masks = pred_masks.cpu()

    n = min(max_show, images.size(0))
    for i in range(n):
        img  = images[i].permute(1, 2, 0).numpy()            # CHW -> HWC
        gt   = gt_masks[i, 0].numpy()
        pred = pred_masks[i, 0].numpy()

        plt.figure(figsize=(12, 4))
        # 原圖
        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(img.astype('uint8'))
        plt.axis('off')

        # 真值 mask
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(gt, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')

        # 預測 mask
        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow(pred, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')

        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"triplet_{i:03d}.png"))
        plt.show()
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--data_path','-p' ,type=str,default='dataset/oxford-iiit-pet', help='path of the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='batch size')
    parser.add_argument('--model_para_path', '-m', type=str, default="epochs_2type_rnet34andU.pth", help='path to the stored model weoght')
    parser.add_argument('--save_img_dir', '-i', type=str, default="pred_mask", help='path to the stored model weoght')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    inference(args)
