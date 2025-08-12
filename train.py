import argparse
from oxford_pet import load_dataset
import torch
import matplotlib.pyplot as plt
from models.unet import UNet
from torch.utils.data import DataLoader
from tqdm import tqdm
from evaluate import evaluate
torch.backends.cudnn.benchmark = True  # 對固定輸入大小加速
def train(args):
    data_path= args.data_path
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    # 1. 準備資料
    train_dataset = load_dataset(data_path,"train")
    valid_dataset = load_dataset(data_path,"valid")
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,              # Windows 也 OK；如果記憶體吃緊可降到 2
        pin_memory=True,
        persistent_workers=True     # 需要 num_workers > 0
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,              # Windows 也 OK；如果記憶體吃緊可降到 2
        pin_memory=True,
        persistent_workers=True     # 需要 num_workers > 0
    )

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 2. 建立模型
    model = UNet(in_channels=3, out_channels=1)
    model = model.to(DEVICE)

    # 3. 損失函數與 optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    # 下面這步驟讓optimzier 會知道哪些para進行update para(weigh, bias)
    # 後面backward()時候 grad 會存到para的grad 
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


    print_every = 10  # 每 10 個 batch 印一次

    best_val_dice = 0.0
    for epoch in range(epochs):
        
        model.train()
        running = 0.0
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        # for batch_idx, batch in enumerate(loader):
        for batch_idx, batch in enumerate(pbar):
            # images = batch["image"].float().to(DEVICE)
            # masks = batch["mask"].float().to(DEVICE)

            images = batch["image"].float().to(DEVICE, non_blocking=True)
            masks  = batch["mask"].float().to(DEVICE, non_blocking=True)
            # forward
            # 建好完整comp graph
            # optimizer.zero_grad()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, masks)

            # loss.backward() 會沿著計算圖往回傳播，把算出來的梯度值存到
            # model.parameters() 回傳的每個參數物件的 .grad 屬性 裡。
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            running += loss.item()
            if (batch_idx + 1) % print_every == 0:
                avg = running / print_every
                pbar.set_postfix({"loss": f"{avg:.4f}"})
                running = 0.0
        val_dice = evaluate(model=model, data_loader=valid_loader,DEVICE=DEVICE)
        # print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(loader):.4f}")
        if val_dice > best_val_dice:
            print(f"dice score:{val_dice}")
            best_val_dice = val_dice
            torch.save(model.state_dict(), "best_unet.pt")
            print("  ✅ New best model saved.")
    # sample = dataset[0]
    # # 因為 SimpleOxfordPetDataset 已經把 image 轉成 (C, H, W)
    # # 要還原成 (H, W, C) 才能用 plt.imshow 顯示
    # image = sample['image'].transpose(1, 2, 0)  # CHW → HWC
    # mask = sample['mask'].squeeze(0)             # (1, H, W) → (H, W)

    # # 顯示原圖和 mask
    # plt.figure(figsize=(8,4))
    # plt.subplot(1,2,1)
    # plt.title("Image")
    # plt.imshow(image)
    # plt.axis('off')

    # plt.subplot(1,2,2)
    # plt.title("Mask")
    # plt.imshow(mask, cmap='gray')
    # plt.axis('off')

    # plt.show()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str,default='dataset/oxford-iiit-pet', help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)