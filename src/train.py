import torch
import torch.nn as nn
import sys
from pix2pix import Pix2Pix
from utils import Optimizer
from torch.optim import lr_scheduler
import time
import copy
from torch.autograd import Variable
from utils import dice_coeff
from data_loader import NYU_Depth_V2
import cv2
import numpy as np
import os
# from torchvision.transforms import ToPILImage

if torch.cuda.is_available():
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
	torch.set_default_tensor_type('torch.FloatTensor')

train_set = NYU_Depth_V2('train')
print('Loaded training set')
val_set = NYU_Depth_V2('val')
print('Loaded val set')

dataset = {0: train_set, 1: val_set}

opt = Optimizer(lr=2 * 1e-4, beta1=0.5, lambda_L1=0.1, n_epochs=50, batch_size=2)


dataloader = {x: torch.utils.data.DataLoader(
    dataset[x], batch_size=opt.batch_size, shuffle=True, num_workers=0) for x in range(2)}

dataset_size = {x: len(dataset[x]) for x in range(2)}

print(dataset_size)
def train(opt, model_name):
	if model_name == 'pix2pix':
		model = Pix2Pix(opt)
		# best_model_wts = copy.deepcopy(model.state_dict())
		# best_acc = 0.0
		for epoch in range(opt.epochs):
			since = time.time()
			print('Epoch ' + str(epoch) + ' running')
			for phase in range(2):
				val_dice = 0
				count = 0
				for i, Data in enumerate(dataloader[phase]):
					inputs, masks = Data
					inputs, masks = Variable(inputs), Variable(masks)
					Data = inputs, masks
					with torch.set_grad_enabled(phase == 0):
						model.get_input(Data)
						if phase == 0:
							model.optimize()
						else:
							pred_mask = model.forward(inputs)
							print(pred_mask.size())

							# t = ToPILImage()
							# a = {j: t(pred_mask[j].cpu().detach()) for j in range(pred_mask.size()[0])}
							# b = {j: t(inputs[j].cpu().detach()) for j in range(inputs.size()[0])}
							for j in range(pred_mask.size()[0]):
								cv2.imwrite( os.path.join('../results/pred_masks', 
									'mask_{}_{}_{}.png'.format(i, j, epoch)),
								  np.array(pred_mask[j].cpu().detach()).reshape(256, 256, 3))
								cv2.imwrite( os.path.join('../results/inputs', 
									'input_{}_{}_{}.png'.format(i, j, epoch)),
								  np.array(inputs[j].cpu().detach()).reshape(256, 256, 3))
							# print(pred_mask.size())
							# print(masks.size())
							# print(dice_coeff(pred_mask, masks))
							val_dice += dice_coeff(pred_mask, masks)
							count += 1
			print("Validation Dice Coefficient is " + str(val_dice/count))
			time_elapsed = time.time() - since
			print('Epoch completed in {:.0f}m {:.0f}s'.format(
        			time_elapsed // 60, time_elapsed % 60))


train(opt, 'pix2pix')