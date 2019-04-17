import torch
import torch.nn as nn
import sys
from pix2pix import Pix2Pix
from cycleGAN import cycleGan
from utils import Optimizer, cgOptimizer
from torch.optim import lr_scheduler
import time
import copy
from torch.autograd import Variable
from utils import dice_coeff, normalization, denormalize
from data_loader import NYU_Depth_V2, NYU_Depth_V2_v2
import cv2
import numpy as np
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pickle

# from torchvision.transforms import ToPILImage

torch.set_default_tensor_type('torch.FloatTensor')
# if torch.cuda.is_available():
# 	torch.set_default_tensor_type('torch.cuda.FloatTensor')
# else:
# 	torch.set_default_tensor_type('torch.FloatTensor')

train_set = NYU_Depth_V2('train')
print('Loaded training set')
val_set = NYU_Depth_V2('val')
print('Loaded val set')



#For Cycle GAN
# image_size = 256
# transform = transforms.Compose([ transforms.Resize(image_size),
#                                     transforms.ToTensor()])
# img_path = '../data/imgs'
# depth_path = '../data/depths'
# train_X_dir= img_path +'_train'
# train_Y_dir = depth_path + '_train'
# val_X_dir = img_path + '_val'
# val_Y_dir = depth_path + '_val'

# class ConcatDataset(torch.utils.data.Dataset):
#     def __init__(self, *datasets):
#         self.datasets = datasets

#     def __getitem__(self, i):
#         return tuple(d[i] for d in self.datasets)

#     def __len__(self):
#         return min(len(d) for d in self.datasets)

# cg_train_loader = torch.utils.data.DataLoader(
#              ConcatDataset(
#                  datasets.ImageFolder(train_X_dir,transform),
#                  datasets.ImageFolder(train_Y_dir,transform)
#              ),
#              batch_size=16, shuffle=True,
#              num_workers=0)

# cg_val_loader = torch.utils.data.DataLoader(
#              ConcatDataset(
#                  datasets.ImageFolder(val_X_dir,transform),
#                  datasets.ImageFolder(val_Y_dir,transform)
#              ),
#              batch_size=16, shuffle=True,
#              num_workers=0)
loadSize = 0
fineSize = 0
if len(sys.argv) == 3:
	loadSize = int(sys.argv[1])
	fineSize = int(sys.argv[2])
	print("LoadSize and FineSize loaded")
	cg_train_set = NYU_Depth_V2_v2('train', loadSize, fineSize)
	print('Loaded training set')
	cg_val_set = NYU_Depth_V2_v2('val', loadSize, fineSize)
	print('Loaded val set')



dataset = {0: train_set, 1: val_set}

if len(sys.argv) == 3:
	cg_dataset = {0: cg_train_set, 1: cg_val_set}

else:
	cg_dataset = {0: train_set, 1: val_set}


opt = Optimizer(lr=1e-4, beta1=0.5, lambda_L1=0.01, n_epochs=100, batch_size=4)

cg_opt = cgOptimizer(input_nc=3, output_nc=3, ngf=64, norm=nn.InstanceNorm2d, no_dropout=True, n_blocks=9,
                 padding_type='replicate', ndf=64, n_layers_D = 3, pool_size = 50, lr = 0.0001, beta1 = 0.5 , lambda_A = 5, lambda_B = 5,pool=False)



dataloader = {x: torch.utils.data.DataLoader(
    dataset[x], batch_size=opt.batch_size, shuffle=True, num_workers=0) for x in range(2)}

cg_train_loader = torch.utils.data.DataLoader(cg_dataset[0], batch_size = 2, shuffle=True, num_workers=0)
cg_val_loader = torch.utils.data.DataLoader(cg_dataset[1], batch_size = 4, shuffle=True, num_workers=0)


dataset_size = {x: len(dataset[x]) for x in range(2)}

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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
					inputs, masks = inputs.to(device), masks.to(device)
					inputs = normalization(inputs)  ##Changes made here
					masks = normalization(masks)
					inputs, masks = Variable(inputs), Variable(masks)  ##Ye Variable kyu likha hai ek baar batana
					Data = inputs, masks                               ## --> kyuki isse computation fast ho jaata
																	   ## memoization ki vajah se
					with torch.set_grad_enabled(phase == 0):
						model.get_input(Data)
						if phase == 0:
							model.optimize()
							# pred_mask = model.forward(inputs)
							# if i%25 == 0:
							# 	for j in range(pred_mask.size()[0]):
							# 		cv2.imwrite( os.path.join('../results/pred_masks',
							# 			'train_mask_{}_{}_{}.png'.format(i, j, epoch)),
							# 		  np.array(denormalize(pred_mask[j]).cpu().detach()).reshape(256, 256, 3))
							# 		cv2.imwrite( os.path.join('../results/inputs',
							# 			'train_input_{}_{}_{}.png'.format(i, j, epoch)),
							# 		  np.array(denormalize(inputs[j]).cpu().detach()).reshape(256, 256, 3))
						else:
							pred_mask = model.forward(inputs)

							# t = ToPILImage()
							# a = {j: t(pred_mask[j].cpu().detach()) for j in range(pred_mask.size()[0])}
							# b = {j: t(inputs[j].cpu().detach()) for j in range(inputs.size()[0])}
							# if i%25 == 0:
							for j in range(pred_mask.size()[0]):
								cv2.imwrite( os.path.join('../results/pred_masks',
									'mask_{}_{}_{}.png'.format(i, j, epoch)),
								  np.array(denormalize(pred_mask[j]).cpu().detach()).reshape(256, 256, 3))
								cv2.imwrite( os.path.join('../results/inputs',
									'input_{}_{}_{}.png'.format(i, j, epoch)),
								  np.array(denormalize(inputs[j]).cpu().detach()).reshape(256, 256, 3))

							val_dice += dice_coeff(denormalize(pred_mask, flag=1), denormalize(masks, flag=1))
							count += 1
			print("Validation Dice Coefficient is " + str(val_dice/count))
			time_elapsed = time.time() - since
			print('Epoch completed in {:.0f}m {:.0f}s'.format(
        			time_elapsed // 60, time_elapsed % 60))


	elif model_name == 'CycleGAN':
		model = cycleGan(cg_opt)
		print_freq = 10
		train_iter = iter(cg_train_loader)
		val_iter = iter(cg_val_loader)
		fixed_X , fixed_Y = val_iter.next()
		fixed_X = normalization(fixed_X).to(device)
		fixed_Y = normalization(fixed_Y).to(device)
		loss_Gl = []
		loss_DXl = []
		loss_DYl = []


		num_batches = len(train_iter)
		for epoch in range(200):
			if epoch == 35:
				model.change_lr(model.opt.lr/2)

			if epoch == 80:
				model.change_lr(model.opt.lr/2)

			if epoch == 130:
				model.change_lr(model.opt.lr/2)


			since = time.time()
			print("Epoch ", epoch ," entering ")
			train_iter = iter(cg_train_loader)
			for batch in range(num_batches):
				print("Epoch ", epoch ,"Batch ", batch, " running with learning rate ", model.opt.lr)
				inputX,inputY = train_iter.next()
				inputX = normalization(inputX).to(device)
				inputY = normalization(inputY).to(device)
				model.get_input(inputX,inputY)
				model.optimize()
				# print("Dx Loss : {:.6f} Dy Loss: {:.6f} Generator Loss: {:.6f} ".format(model.dx_loss, model.dy_loss, model.gen_loss))
				print("Model dx loss " , float(model.loss_D_X), "Model dy loss", float(model.loss_D_Y), "model_gen_loss", float(model.loss_G))



			if (epoch+1)%10 == 0:
				# torch.set_grad_enabled(False)
				depth_map = model.G_XtoY.forward(fixed_X)
				for j in range(depth_map.size()[0]):
					if cg_opt.n_blocks==6:
						cv2.imwrite( os.path.join('../cgresults/pred_masks',
							'mask_{}_{}_{}.png'.format(batch, j, epoch)),
						  np.array(denormalize(depth_map[j]).cpu().detach()).reshape(256,256,3))
						if epoch == 9:
							cv2.imwrite( os.path.join('../cgresults/inputs',
								'input_{}_{}_{}.png'.format(batch, j, epoch)),
							  np.array(denormalize(fixed_X[j]).cpu().detach()).reshape(256,256, 3))
					else:
						cv2.imwrite( os.path.join('../cgresults/r-9-pred_masks',
									'mask_{}_{}_{}.png'.format(batch, j, epoch)),
								  np.array(denormalize(depth_map[j]).cpu().detach()).reshape(256,256,3))
						if epoch == 9:
							cv2.imwrite( os.path.join('../cgresults/r-9-inputs',
								'input_{}_{}_{}.png'.format(batch, j, epoch)),
							  np.array(denormalize(fixed_X[j]).cpu().detach()).reshape(256,256, 3))

				# torch.set_grad_enabled(True)

			print("Time to finish epoch ", time.time()-since)

			torch.save(model, '../CGmodel/best_model5.pt')
			loss_Gl.append(float(model.loss_G))
			loss_DXl.append(float(model.loss_D_X))
			loss_DYl.append(float(model.loss_D_Y))
			with open('../CGloss/lossG5.pk', 'wb') as f:
				pickle.dump(loss_Gl, f)
			with open('../CGloss/lossD_X5.pk', 'wb') as f:
				pickle.dump(loss_DXl, f)
			with open('../CGloss/lossd_Y5.pk', 'wb') as f:
				pickle.dump(loss_DYl, f)



train(opt, 'CycleGAN')
