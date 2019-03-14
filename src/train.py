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

if torch.cuda.is_available():
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
	torch.set_default_tensor_type('torch.FloatTensor')

data = None ###### Load data

output = None ######## Load output

def train(opt, model_name):
	since = time.time()
	if model_name == 'pix2pix':
		model = Pix2Pix(opt)
		best_model_wts = copy.deepcopy(model.state_dict())
		best_acc = 0.0
		for epoch in range(opt.epochs):
			print('Epoch ' + str(epoch) + ' running')
			for phase in range(2):
				val_dice = 0
            	count = 0
				for i, Data in enumerate(dataloader[phase]):
					inputs, masks = Data
					inputs, masks = Variable(inputs), Variable(masks)
					Data = Variable(Data)
					with torch.set_grad_enabled(phase == 0):
						Pix2Pix.get_input(Data)
						if phase == 0:
							Pix2Pix.optimize()
						else:
							pred_mask = Pix2Pix.forward(inputs)
							val_dice += dice_coeff(pred_mask, masks)
			print("Validation Dice Coefficient is " + str(val_dice))





	