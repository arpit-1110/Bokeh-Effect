import torch
import numpy as np
import torch.utils.data
import glob
import cv2
import random

class NYU_Depth_V2(torch.utils.data.Dataset):
    def __init__(self, data_type, transform=None):
        img_path = '../data/imgs'
        depth_path = '../data/depths'
        if data_type != 'train' and data_type != 'val' and data_type != 'test':
            raise ValueError('Invalid data type')
        self.data_type = data_type
        self.X_train = sorted(glob.glob(img_path + '_train/*img.png'))
        self.y_train = sorted(glob.glob(depth_path + '_train/*depth.png'))
        self.X_val = sorted(glob.glob(img_path + '_val/*img.png'))  #Removed [:100] from here
        self.y_val = sorted(glob.glob(depth_path + '_val/*depth.png'))
        self.X_test = sorted(glob.glob(img_path + '_test/*img.png'))
        self.y_test = sorted(glob.glob(depth_path + '_test/*depth.png'))

    def __len__(self):
        if self.data_type == 'train':
            return len(self.X_train)
        if self.data_type == 'val':
            return len(self.X_val)
        if self.data_type == 'test':
            return len(self.X_test)

    def __getitem__(self, idx):
        if self.data_type == 'train':
            if self.X_train[idx][12:-7] == self.y_train[idx][14:-9]:
                X = np.array(cv2.imread(self.X_train[idx])).reshape(3, 256, 256)
                y = np.array(cv2.imread(self.y_train[idx])).reshape(3, 256, 256)

            else:
                raise ValueError

            return torch.from_numpy(X).float(), torch.from_numpy(y).float()

        if self.data_type == 'val':
            if self.X_val[idx][12:-7] == self.y_val[idx][14:-9]:
                X = np.array(cv2.imread(self.X_val[idx])).reshape(3, 256, 256)
                y = np.array(cv2.imread(self.y_val[idx])).reshape(3, 256, 256)

            else:
                raise ValueError

            return torch.from_numpy(X).float(), torch.from_numpy(y).float()

        if self.data_type == 'test':
            if self.X_test[idx][12:-7] == self.y_test[idx][14:-9]:
                X = np.array(cv2.imread(self.X_test[idx])).reshape(3, 256, 256)
                y = np.array(cv2.imread(self.y_test[idx])).reshape(3, 256, 256)

            else:
                raise ValueError

            return torch.from_numpy(X).float(), torch.from_numpy(y).float()


class NYU_Depth_V2_v2(torch.utils.data.Dataset):
    def __init__(self, data_type, loadsize, finesize, transform=None):
        img_path = '../data/imgs'
        depth_path = '../data/depths'
        self.loadsize = loadsize
        self.finesize = finesize
        if data_type != 'train' and data_type != 'val' and data_type != 'test':
            raise ValueError('Invalid data type')
        self.data_type = data_type
        self.X_train = sorted(glob.glob(img_path + '_train/*img.png'))
        self.y_train = sorted(glob.glob(depth_path + '_train/*depth.png'))
        self.X_val = sorted(glob.glob(img_path + '_val/*img.png'))  #Removed [:100] from here
        self.y_val = sorted(glob.glob(depth_path + '_val/*depth.png'))
        self.X_test = sorted(glob.glob(img_path + '_test/*img.png'))
        self.y_test = sorted(glob.glob(depth_path + '_test/*depth.png'))

    def __len__(self):
        if self.data_type == 'train':
            return len(self.X_train)
        if self.data_type == 'val':
            return len(self.X_val)
        if self.data_type == 'test':
            return len(self.X_test)

    def __getitem__(self, idx):
        if self.data_type == 'train':
            if self.X_train[idx][12:-7] == self.y_train[idx][14:-9]:
                imgX = cv2.resize(cv2.imread(self.X_train[idx]), (self.loadsize,self.loadsize), interpolation=cv2.INTER_CUBIC)
                imgy = cv2.resize(cv2.imread(self.y_train[idx]), (self.loadsize,self.loadsize), interpolation=cv2.INTER_CUBIC)
                a = random.randint(0,self.loadsize-self.finesize-2)
                # print('Random call hua hai')
                b = random.randint(0,self.loadsize-self.finesize-2)
                imgX = imgX[a:a+self.finesize,b:b+self.finesize]
                imgy = imgy[a:a+self.finesize,b:b+self.finesize]
                X = np.array(imgX).reshape(3,self.finesize,self.finesize)
                y = np.array(imgy).reshape(3,self.finesize,self.finesize)

            else:
                raise ValueError

            return torch.from_numpy(X).float(), torch.from_numpy(y).float()

        if self.data_type == 'val':
            if self.X_val[idx][12:-7] == self.y_val[idx][14:-9]:

                X = np.array(cv2.imread(self.X_val[idx])).reshape(3, 256, 256)
                y = np.array(cv2.imread(self.y_val[idx])).reshape(3, 256, 256)  # Changes mad
            else:
                raise ValueError

            return torch.from_numpy(X).float(), torch.from_numpy(y).float()

        if self.data_type == 'test':
            if self.X_test[idx][12:-7] == self.y_test[idx][14:-9]:

                X = np.array(cv2.imread(self.X_val[idx])).reshape(3, 256, 256)
                y = np.array(cv2.imread(self.y_val[idx])).reshape(3, 256, 256)  # Changes made in the above lines
            else:
                raise ValueError

            return torch.from_numpy(X).float(), torch.from_numpy(y).float()
