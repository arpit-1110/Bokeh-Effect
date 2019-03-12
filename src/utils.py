import torch
import sys

class Optimizer():
	def __init__(self, lr=None, mu=None, beta1=None, weight_decay=None,
				 lambda_L1=None):
		self.lr = lr
		self.mu = mu
		self.beta1 = beta1
		self.w_decay = weight_decay
		self.lambda_L1 = lambda_L1