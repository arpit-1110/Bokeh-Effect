import torch
import torch.nn as nn
import torch.optim as optim
from cnn_utils import Generator
from cnn_utils import Discriminator


if torch.cuda.is_available():
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
	torch.set_default_tensor_type('torch.FloatTensor')

class Pix2Pix():
	def __init__(self, opt):
		super(Pix2Pix, self).__init__()
		self.gen = Generator()
		self.dis = Discriminator()
		self.GAN_loss = nn.MSELoss()
		self.L1 = nn.L1Loss()
		self.opG = optim.Adam(self.gen.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
		self.opD = optim.Adam(self.dis.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
		self.opt = opt

	def get_input(self, data):
		self.img, self.true_seg = data

	def forward(self, x=None):
		if x is None:
			self.pred_seg = self.gen(self.img)
		else :
			return self.gen(x)

	def backwardD(self):
		pred_gen_D = self.dis(self.img.detach(), self.pred_seg.detach())
		self.loss1 = self.GAN_loss(pred_gen_D, torch.ones(pred_gen_D.size()))
		predD = self.dis(self.img, self.true_seg)
		self.loss2 = self.GAN_loss(predD, torch.zeros(predD.size()))

		self.loss = 0.5 * (self.loss1 + self.loss2)
		self.loss.backward()

	def backwardG(self):
		pred_gen_D = self.dis(self.img, self.pred_seg)
		self.loss1 = self.GAN_loss(pred_gen_D, torch.ones(pred_gen_D.size()))
		self.loss2 = self.opt.lambda_L1 * self.L1(self.pred_seg, self.true_seg)

		self.loss = self.loss1 + self.loss2
		self.loss.backward()

	def set_requires_grad(self, model, grad=False):
		if not isinstance(model, list):
			model = [model]

		for x in model:
			if x is not None:
				for param in x.parameters():
					param.requires_grad = grad

	def optimize(self):  
		self.forward()
		self.set_requires_grad(self.dis, True)
		self.opD.zero_grad()
		self.backwardD()
		self.opD.step()
		self.set_requires_grad(self.dis, False)
		self.opG.zero_grad()
		self.backwardG()
		self.opG.step()


