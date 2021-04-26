import numpy as np

# dice value
def dice_coef(pred_label, gt_label):
    # list of classes
    c_list = np.unique(gt_label)

    dice_c = []
    for c in range(1,len(c_list)): # dice not for bg
        # intersection
        ints = np.sum(((pred_label == c_list[c]) * 1) * ((gt_label == c_list[c]) * 1))
        # sum
        sums = np.sum(((pred_label == c_list[c]) * 1) + ((gt_label == c_list[c]) * 1)) + 0.0001
        dice_c.append((2.0 * ints) / sums)

    return dice_c

def dice_coef_bi(pred_label, gt_label):
    pred_label = pred_label.flatten()
    gt_label = gt_label.flatten()
    dice_c = []
    # intersection
    ints = np.sum(((pred_label == 1) * 1) * ((gt_label == 1) * 1))
    # sum
    sums = np.sum(((pred_label == 1) * 1) + ((gt_label == 1) * 1)) + 0.0001
    dice_c.append((2.0 * ints) / sums)

    return dice_c


# dice value
def soft_dice(input_, target):
    smooth = 1.

    input_flat = input_.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = input_flat * target_flat

    dice = (2 * intersection.sum() + smooth) / (input_flat.sum() + target_flat.sum() + smooth)

    return dice