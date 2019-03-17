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
        # print(intersection)
        # print(iflat.sum())
        # print(tflat.sum())
        coeff += (2. * intersection) / (iflat.sum() + tflat.sum() + eps)
    return coeff / (inputs.shape[0])

def normalization(X):
    return X / 127.5 - 1.0

def denormalize(X, flag=None):
	if flag is None:
		return (X + 1.0) * 127.5
	else:
		return (X + 1.0) / 2.0

def dice_loss(inputs, target):
    return 1 - dice_coeff(inputs, target)

class Optimizer():
	def __init__(self, lr=None, mu=None, beta1=None, weight_decay=None,
				 lambda_L1=None, n_epochs=None, scheduler=None, batch_size=None):
		self.lr = lr
		self.mu = mu
		self.beta1 = beta1
		self.w_decay = weight_decay
		self.lambda_L1 = lambda_L1
		self.epochs = n_epochs
		self.scheduler = scheduler
		self.batch_size = batch_size
