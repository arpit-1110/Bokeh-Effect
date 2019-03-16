import torch
import torch.nn as nn
import sys
from pix2pix import Pix2Pix
from utils import Optimizer
from torch.optim import lr_scheduler
import time
import copy
from torch.autograd import Variable
from utils import dice_coeff, normalization
from data_loader import NYU_Depth_V2
from PIL import Image

if torch.cuda.is_available():
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
	torch.set_default_tensor_type('torch.FloatTensor')

train_set = NYU_Depth_V2('train')
print('Loaded training set')
val_set = NYU_Depth_V2('val')
print('Loaded val set')

dataset = {0: train_set, 1: val_set}

opt = Optimizer(lr=1e-2, mu=0.9, beta1=0.9, lambda_L1=1, n_epochs=10, batch_size=4)


dataloader = {x: torch.utils.data.DataLoader(
    dataset[x], batch_size=opt.batch_size, shuffle=True, num_workers=0)for x in range(2)}

dataset_size = {x: len(dataset[x]) for x in range(2)}

# print(dataset_size)
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
					inputs = normalization(inputs)  ##Changes made here
					masks = normalization(masks)
					inputs, masks = Variable(inputs), Variable(masks)  ##Ye Variable kyu likha hai ek baar batana
					Data = inputs, masks
					with torch.set_grad_enabled(phase == 0):
						model.get_input(data=Data)
						if phase == 0:
							model.optimize()
						else:
							pred_mask = model.forward(inputs)
							t = ToPILImage()  ## Arpit is line me kuch error hai
							a = {j: t(pred_mask[i].cpu().detach()) for j in range(opt.batch_size)}
							b = {j: t(inputs[i].cpu().detach()) for j in range(opt.batch_size)}
							for j in range(opt.batch_size):
								a[j].save('../results/pred_masks/mask' +
								 '_' + str(i) + '_' + str(j) + '_' + str(epoch) + '.png')
								b[j].save('../results/inputs/input' +
								 '_' + str(i) + '_' + str(j) + '_' + str(epoch) + '.png')
							val_dice += dice_coeff(pred_mask, masks)
			print("Validation Dice Coefficient is " + str(val_dice))
			time_elapsed = time.time() - since
			print('Epoch completed in {:.0f}m {:.0f}s'.format(
        			time_elapsed // 60, time_elapsed % 60))


train(opt, 'pix2pix')
