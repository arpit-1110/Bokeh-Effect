import torch
import sys
import math


def dice_coeff(inputs, target):
    eps = 1e-7
    coeff = 0
    for i in range(inputs.shape[0]):
        iflat = inputs[i, :, :, :].view(-1)
        tflat = target[i, :, :, :].view(-1)
        intersection = torch.dot(iflat, tflat)
        coeff += (2. * intersection) / (iflat.sum() + tflat.sum() + eps)
    return coeff / inputs.shape[0]


def dice_loss(inputs, target):
    return 1 - dice_coeff(inputs, target)

class Optimizer():
	def __init__(self, lr=None, mu=None, beta1=None, weight_decay=None,
				 lambda_L1=None, n_epochs=None, scheduler=None):
		self.lr = lr
		self.mu = mu
		self.beta1 = beta1
		self.w_decay = weight_decay
		self.lambda_L1 = lambda_L1
		self.epochs = no_epochs
		self.scheduler = scheduler
