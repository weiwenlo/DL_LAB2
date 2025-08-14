import matplotlib.pyplot as plt
import os
import numpy as np

def dice_score(pred_mask, gt_mask,eps=1e-7):
    dice_sum = 0
    # implement the Dice score here
    inter = (pred_mask * gt_mask).sum(dim=(1,2,3))         # per-image
    psum  = pred_mask.sum(dim=(1,2,3))
    gsum  = gt_mask.sum(dim=(1,2,3))#inter 的 shape 是 (batch_size,)
                                    #psum、gsum 也是 (batch_size,)

    dice  = (2*inter + eps ) / (psum + gsum +eps )  # per-image dice

    # dice.sum() → 把 (batch_size,) 的向量加總成一個 scalar tensor，shape 從 (B,) 變成 ()。
    # 例如 tensor(3.4567, device='cuda:0')
    dice_sum += dice.sum().item()
    return  dice_sum

def plot_accuracy(train_acc_list, valid_acc_list, filename, results_dir='plot'):

    num_epochs = len(train_acc_list)
    fig = plt.figure()
    plt.figure()
    plt.plot(np.arange(1, num_epochs+1),
             train_acc_list, label='Training')
    plt.plot(np.arange(1, num_epochs+1),
             valid_acc_list, label='Validation')
    for x, y in zip(np.arange(1, num_epochs+1), train_acc_list):
        plt.text(x, y, f"{y:.4f}", ha='center', va='bottom')

    for x, y in zip(np.arange(1, num_epochs+1), valid_acc_list):
        plt.text(x, y, f"{y:.4f}", ha='center', va='bottom')

    plt.xlabel('Epoch')
    plt.ylabel('Dice score')
    plt.legend()

    plt.tight_layout()

    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)  # 自動建立資料夾（已存在就跳過）
        image_path = os.path.join(
            results_dir, filename+'.pdf')
        plt.savefig(image_path)
    plt.show()
    plt.close(fig)  # 關掉這張圖，避免下次殘留