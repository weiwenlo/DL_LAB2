import argparse
from oxford_pet import load_dataset
import torch
from models.unet import UNet
from models.resnet34_unet import ResNetU,BasicBlock
from torch.utils.data import DataLoader
from tqdm import tqdm
from evaluate import evaluate
from utils import *
import random

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train(args):
    data_path= args.data_path
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate 
    comment = args.comment
    model_type = args.model_type
    trainfile_name = "e_"+str(epochs)+"_"+"t_"+model_type+"lr_"+str(args.learning_rate)+"_bs_"+str(batch_size)+"_"+comment 
    print(trainfile_name+".pth")
    # 1. 準備資料
    train_dataset = load_dataset(data_path,"train")
    valid_dataset = load_dataset(data_path,"valid")
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=4,              # Windows 也 OK；如果記憶體吃緊可降到 2
    #     pin_memory=True,
    #     persistent_workers=True     # 需要 num_workers > 0
    # )
    g = torch.Generator()
    g.manual_seed(123)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
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
    if model_type =="unet":
        model = UNet(in_channels=3, out_channels=1)
    elif model_type =="rnet34andU":
        model = ResNetU(BasicBlock, layers=[3, 4, 6, 3], num_classes=1)
    model = model.to(DEVICE)

    # 3. 損失函數與 optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    # 下面這步驟讓optimzier 會知道哪些para進行update para(weigh, bias)
    # 後面backward()時候 grad 會存到para的grad 
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


    # print_every = 10  # 每 10 個 batch 印一次
    train_acc_list = []
    valid_acc_list = []
    best_val_dice = 0.0
    patience = 5                # 最多允許連續幾個 epoch 沒改善
    no_improve_count = 0         # 累積沒改善的次數
    for epoch in range(epochs):
        
        model.train()
        running = 0.0
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        # pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
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
            #追蹤loss 用
            # if (batch_idx + 1) % print_every == 0:
            #     avg = running / print_every
            #     pbar.set_postfix({"loss": f"{avg:.4f}"})
            #     running = 0.0
        train_dice = evaluate(model=model, data_loader=train_loader,DEVICE=DEVICE)
        val_dice = evaluate(model=model, data_loader=valid_loader,DEVICE=DEVICE)
        train_acc_list.append(train_dice)
        valid_acc_list.append(val_dice)
        # print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(loader):.4f}")
        if val_dice > best_val_dice:
            print(f"dice score:{val_dice}")
            best_val_dice = val_dice
            no_improve_count = 0

            torch.save(model.state_dict(), trainfile_name+".pth")
            print("  ✅ New best model saved.")
        else:
            no_improve_count += 1
        if no_improve_count >= patience:
            print("Early stopping triggered.")
            break
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
    return train_acc_list, valid_acc_list, trainfile_name

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str,default='dataset/oxford-iiit-pet', help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=2, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')
    # parser.add_argument('--model_para_path', '-m', type=str, default="best_unet.pth", help='para path')
    parser.add_argument('--model_type', '-t', type=str, default="rnet34andU", help='model type')
    parser.add_argument('--comment', '-c', type=str, default="flip05", help='model type')
    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    seed = 123
    torch.cuda.manual_seed_all(seed)# 多組GPU需要固定
    torch.manual_seed(seed) # CPU和GPU固定
    np.random.seed(seed)
    # 關掉 benchmark，避免根據輸入大小選不同 kernel
    # torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True) #演算法用非隨機性
    train_acc_list, valid_acc_list,trainfile_name= train(args)
    plot_accuracy(train_acc_list, valid_acc_list,trainfile_name)