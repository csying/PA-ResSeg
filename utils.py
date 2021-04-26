import shutil
import torch
import sys
import os
import numpy as np
from PIL import Image

def save_checkpoint(state, is_best,fold,filename):
    ckpt_num_up = 10
    if is_best:
        torch.save(state, os.path.join(fold, filename))
        print('save model ', filename)
    ckpts = os.listdir(fold)
    ckpt_nums = len(ckpts)
    if ckpt_nums > ckpt_num_up:
        dices = [float(ckpt.split('-')[-1][:-4]) for ckpt in ckpts]
        min_dice = min(dices)
        remove_name = ckpts[dices.index(min_dice)]
        try:
            os.remove(os.path.join(fold, remove_name))
            print('remove ', remove_name)
        except:
            print('Eception')


def save_checkpoint_new(state, is_best,fold,filename,SAVE=False):
    ckpt_num_up = 12
    if is_best:
        torch.save(state, os.path.join(fold, filename))
        print('save model ', filename)
    elif SAVE:
        torch.save(state, os.path.join(fold, filename))
        print('save model in specific epoch ', filename)

    # ckpts = os.listdir(fold)
    # ckpt_nums = len(ckpts)
    # if ckpt_nums > ckpt_num_up:
    #     dices = [float(ckpt.split('-')[-1][:-4]) for ckpt in ckpts]
    #     min_dice = min(dices)
    #     remove_name = ckpts[dices.index(min_dice)]
    #     try:
    #         os.remove(os.path.join(fold, remove_name))
    #         print('remove ', remove_name)
    #     except:
    #         print('Eception')


def get_gt_labels(data_dir):
    files = os.listdir(data_dir)
    files.sort()
    gt_labels = []
    for file in files:
        label = Image.open(os.path.join(data_dir, file))
        label = torch.tensor(np.asarray(label, dtype=np.uint8) / 255, dtype=torch.uint8)
        gt_labels.append(label)
    arr_gt_labels = torch.cat(gt_labels, dim=0)
    print('load gt labels done!')
    return arr_gt_labels

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

