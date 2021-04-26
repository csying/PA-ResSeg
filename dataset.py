import numpy as np
import os
import torch

import cv2
import random
from PIL import Image

from torch.utils.data import Dataset
from skimage.io import imsave,imshow,imread

def load_image(file):
    return Image.open(file)

class PNGDataset(Dataset):
    def __init__(self, data_path, batch_names,ins = 224):
        self.data_path = data_path

        # get all filenames and corresponding batch name
        self.filenames,self.foldernames = self.get_file_names(batch_names)

        self.ins = ins

        # for debug
        self.sample_count = 0

    def get_file_names(self,batches):
        train_file_names = []
        corres_batch = []

        for batch in batches:
            names = []
            files = os.listdir(os.path.join(self.data_path, batch, 'mask/'))
            files.sort()
            for i in range(len(files)):
                names.append(batch)
            # files = files[:50]  # for debug
            train_file_names += files
            corres_batch += names

        print('Number of samples: ',len(train_file_names))
        return train_file_names,corres_batch

    def crop_samples(self,file,tI,lI):
        # select sample from liverROI
        txt_path = '/Project/LiverBox/'
        caseNum = file[3:6]
        maxmin = np.loadtxt(os.path.join(txt_path, 'box_' + caseNum + '.txt', ),
                            delimiter=' ')  # liver bounding box顶点坐标
        minindex = maxmin[0:3]  # y,x,z(corresponding to the real images people see)
        maxindex = maxmin[3:6]
        minx = int(minindex[1])  # min x of liver ROI
        maxx = int(maxindex[1])  # max x of liver ROI
        miny = int(minindex[0])  # min y of liver ROI
        maxy = int(maxindex[0])  # max y of liver ROI
        stx = random.randint(minx, max(minx, maxx - self.ins))
        sty = random.randint(miny, max(miny, maxy - self.ins))

        train_sample = tI[:, stx:stx + self.ins, sty:sty + self.ins]  # sample in liver ROI
        label_sample = lI[stx:stx + self.ins, sty:sty + self.ins]
        nrotate = random.randint(0, 3)
        train_sample = np.rot90(train_sample, nrotate,axes=(1,2))
        label_sample = np.round(np.rot90(label_sample, nrotate,axes=(0,1)))
        nflip = random.randint(0, 1)
        if nflip:
            # print('flip')
            # CHW
            train_sample = train_sample[:, ::-1]
            label_sample = label_sample[::-1]
            # HWC
            #train_sample = np.fliplr(train_sample)
            #label_sample = np.fliplr(label_sample)
        #print('rotate ',nrotate,'flip',nflip)
        return train_sample, label_sample

    def __getitem__(self, index):
        filename = self.filenames[index]
        batchname = self.foldernames[index]

        #=============cv2====================
        # load image sample and label sample
        '''image_sample = cv2.imread(os.path.join(self.data_path,batchname,'img',filename))
        #print('image_samples ', image_sample.max(), image_sample.min())
        image_sample = np.asarray(np.transpose(image_sample / 255.0, [2, 0, 1]), np.float32)  # 0.0-1.0

        label_sample = cv2.imread(os.path.join(self.data_path,batchname,'mask',filename),cv2.IMREAD_GRAYSCALE)
        #print('label_samples ', label_sample.max(), label_sample.min())
        label_sample = np.asarray(label_sample/255, dtype=np.uint8)  # 0,1
        '''
        # =============PIL====================
        with open(os.path.join(self.data_path,batchname,'img',filename), 'rb') as f:
            image_sample = load_image(f).convert('RGB')
        with open(os.path.join(self.data_path,batchname,'mask',filename), 'rb') as f:
            label_sample = load_image(f).convert('L')
        image_sample = np.transpose(np.asarray(image_sample,np.float32) / 255.0, [2, 0, 1])
        label_sample = np.asarray(np.array(label_sample)/255,dtype=np.uint8)
        '''print('image_samples ', image_sample.max(), image_sample.min())
        print('label_samples ', label_sample.max(), label_sample.min())'''

        # crop image and label
        image,label = self.crop_samples(filename,image_sample,label_sample)  # float32[0.0-1.0],uint8(0,1)

        return torch.as_tensor(image.copy(), dtype=torch.float32), torch.as_tensor(label.copy(), dtype=torch.long)
        
    def __len__(self):
        return len(self.filenames)


class Multi_PNGDataset(Dataset):
    def __init__(self, data_path, batch_names, ins=224):
        self.pv_data_path = data_path
        self.art_data_path = data_path.replace('PV','ART')

        # get all filenames and corresponding batch name
        self.filenames, self.foldernames = self.get_file_names(batch_names)

        self.ins = ins

    def get_file_names(self, batches):
        train_file_names = []
        corres_batch = []

        for batch in batches:
            names = []
            files = os.listdir(os.path.join(self.pv_data_path, batch, 'mask/'))
            files.sort()
            for i in range(len(files)):
                names.append(batch)
            train_file_names += files
            corres_batch += names

        print('Number of samples: ', len(train_file_names))
        return train_file_names, corres_batch

    def crop_samples(self, file, tI_pv, tI_art, lI):
        # select sample from liverROI
        txt_path = '/Project/LiverBox/'
        caseNum = file[3:6]
        maxmin = np.loadtxt(os.path.join(txt_path, 'box_' + caseNum + '.txt', ),
                            delimiter=' ')  # liver bounding box顶点坐标
        minindex = maxmin[0:3]  # y,x,z(corresponding to the real images people see)
        maxindex = maxmin[3:6]
        minx = int(minindex[1])  # min x of liver ROI
        maxx = int(maxindex[1])  # max x of liver ROI
        miny = int(minindex[0])  # min y of liver ROI
        maxy = int(maxindex[0])  # max y of liver ROI
        stx = random.randint(minx, max(minx, maxx - self.ins))
        sty = random.randint(miny, max(miny, maxy - self.ins))

        train_sample_pv = tI_pv[:, stx:stx + self.ins, sty:sty + self.ins]  # sample in liver ROI
        train_sample_art = tI_art[:, stx:stx + self.ins, sty:sty + self.ins]  # sample in liver ROI
        label_sample = lI[stx:stx + self.ins, sty:sty + self.ins]
        nrotate = random.randint(0, 3)
        train_sample_pv = np.rot90(train_sample_pv, nrotate, axes=(1, 2))
        train_sample_art = np.rot90(train_sample_art, nrotate, axes=(1, 2))
        label_sample = np.round(np.rot90(label_sample, nrotate, axes=(0, 1)))
        nflip = random.randint(0, 1)
        if nflip:
            # print('flip')
            # CHW
            train_sample_pv = train_sample_pv[:, ::-1]
            train_sample_art = train_sample_art[:, ::-1]
            label_sample = label_sample[::-1]
            # HWC
            # train_sample = np.fliplr(train_sample)
            # label_sample = np.fliplr(label_sample)
        # print('rotate ',nrotate,'flip',nflip)
        return train_sample_pv, train_sample_art, label_sample

    def __getitem__(self, index):
        filename = self.filenames[index]
        batchname = self.foldernames[index]

        # =============PIL====================
        with open(os.path.join(self.pv_data_path, batchname, 'img', filename), 'rb') as f:
            image_sample_pv = load_image(f).convert('RGB')
        with open(os.path.join(self.art_data_path, batchname, 'img', filename), 'rb') as f:
            image_sample_art = load_image(f).convert('RGB')
        with open(os.path.join(self.pv_data_path, batchname, 'mask', filename), 'rb') as f:
            label_sample = load_image(f).convert('L')
        image_sample_pv = np.transpose(np.asarray(image_sample_pv, np.float32) / 255.0, [2, 0, 1])
        image_sample_art = np.transpose(np.asarray(image_sample_art, np.float32) / 255.0, [2, 0, 1])
        label_sample = np.asarray(np.array(label_sample) / 255, dtype=np.uint8)

        # crop image and label
        image_pv, image_art, label = self.crop_samples(filename, image_sample_pv, image_sample_art, label_sample)  # float32[0.0-1.0],uint8(0,1)

        return torch.as_tensor(image_pv.copy(), dtype=torch.float32),torch.as_tensor(image_art.copy(), dtype=torch.float32), torch.as_tensor(label.copy(), dtype=torch.long)

    def __len__(self):
        return len(self.filenames)

class Multi_PNGDataset_val(Dataset):
    def __init__(self, data_path, batch_names, ins=224):
        self.pv_data_path = data_path
        self.art_data_path = data_path.replace('PV','ART')

        # get all filenames and corresponding batch name
        self.filenames, self.foldernames = self.get_file_names(batch_names)

        self.ins = ins

    def get_file_names(self, batches):
        train_file_names = []
        corres_batch = []

        for batch in batches:
            names = []
            files = os.listdir(os.path.join(self.pv_data_path, batch, 'mask/'))
            files.sort()
            for i in range(len(files)):
                names.append(batch)
            # files = files[:50]  # for debug
            train_file_names += files
            corres_batch += names

        print('Number of samples: ', len(train_file_names))
        return train_file_names, corres_batch

    def __getitem__(self, index):
        filename = self.filenames[index]
        batchname = self.foldernames[index]

        # =============PIL====================
        with open(os.path.join(self.pv_data_path, batchname, 'img', filename), 'rb') as f:
            image_sample_pv = load_image(f).convert('RGB')
        with open(os.path.join(self.art_data_path, batchname, 'img', filename), 'rb') as f:
            image_sample_art = load_image(f).convert('RGB')
        with open(os.path.join(self.pv_data_path, batchname, 'mask', filename), 'rb') as f:
            label_sample = load_image(f).convert('L')
        image_sample_pv = np.transpose(np.asarray(image_sample_pv, np.float32) / 255.0, [2, 0, 1])
        image_sample_art = np.transpose(np.asarray(image_sample_art, np.float32) / 255.0, [2, 0, 1])
        label_sample = np.asarray(np.array(label_sample) / 255, dtype=np.uint8)

        return torch.as_tensor(image_sample_pv.copy(), dtype=torch.float32),torch.as_tensor(image_sample_art.copy(), dtype=torch.float32), torch.as_tensor(label_sample.copy(), dtype=torch.long)

    def __len__(self):
        return len(self.filenames)