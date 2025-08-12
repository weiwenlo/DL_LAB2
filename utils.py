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

