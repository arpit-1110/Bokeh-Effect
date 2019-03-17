import torch
import numpy as np
import torch.utils.data
import glob
import cv2

class NYU_Depth_V2(torch.utils.data.Dataset):
	def __init__(self, data_type, transform=None):
		img_path = '../data/imgs'
		depth_path = '../data/depths' 
		if data_type != 'train' and data_type != 'val' and data_type != 'test':
			raise ValueError('Invalid data type')
		self.data_type = data_type 
		self.X_train = glob.glob(img_path + '_train/8*img.jpg')
		self.y_train = glob.glob(depth_path + '_train/8*depth.jpg')
		self.X_val = glob.glob(img_path + '_val/100*img.jpg')
		self.y_val = glob.glob(depth_path + '_val/100*depth.jpg')
		self.X_test = glob.glob(img_path + '_test/*img.jpg')
		self.y_test = glob.glob(depth_path + '_test/*depth.jpg')

	def __len__(self):
		if self.data_type == 'train':
			return len(self.X_train)
		if self.data_type == 'val':
			return len(self.X_val)
		if self.data_type == 'test':
			return len(self.X_test)

	def __getitem__(self, idx):
		if self.data_type == 'train':
			X = np.array(cv2.imread(self.X_train[idx])).reshape(3, 256, 256)
			y = np.array(cv2.imread(self.y_train[idx])).reshape(3, 256, 256)

			return torch.from_numpy(X).float(), torch.from_numpy(y).float()

		if self.data_type == 'val':
			X = np.array(cv2.imread(self.X_val[idx])).reshape(3, 256, 256)
			y = np.array(cv2.imread(self.y_val[idx])).reshape(3, 256, 256)

			return torch.from_numpy(X).float(), torch.from_numpy(y).float()

		if self.data_type == 'test':
			X = np.array(cv2.imread(self.X_test[idx])).reshape(3, 256, 256)
			y = np.array(cv2.imread(self.y_test[idx])).reshape(3, 256, 256)

			return torch.from_numpy(X).float(), torch.from_numpy(y).float()

		

