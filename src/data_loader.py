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


