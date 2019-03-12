import torch
import torch.nn as nn
import sys
from pix2pix import Pix2Pix
from utils import Optimizer

if torch.cuda.is_available():
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
	torch.set_default_tensor_type('torch.FloatTensor')

data = None ###### Load data


def train(opt, model_name):
	if model_name == 'pix2pix':
		model = Pix2Pix(opt)


	