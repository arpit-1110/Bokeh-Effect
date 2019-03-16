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
		self.opG = optim.Adam(self.gen.paramaters(), lr=opt.lr, betas=(opt.beta1, 0.999))
		self.opD = optim.Adam(self.dis.paramaters(), lr=opt.lr, betas=(opt.beta1, 0.999))

	def get_input(self, data):
		self.img = data['0']
		self.true_seg = data['1']

	def forward(self):
		self.pred_seg = self.gen(self.img)

	def backwardD(self):
		pred_gen_D = self.dis(self.img.detach(), self.pred_seg.detach())
		self.loss1 = self.GAN_loss(pred_gen_D, False)
		predD = self.dis(self.img, self.true_seg)
		self.loss2 = self.GAN_loss(predD, True)

		self.loss = 0.5 * (self.loss1 + self.loss2)
		self.loss.backward()

	def backwardG(self):
		pred_gen_D = self.dis(self.img, self.pred_seg)
		self.loss1 = self.GAN_loss(pred_gen_D, True)
		self.loss2 = opt.lambda_L1 * self.L1(self.pred_seg, self.true_seg)

		self.loss = self.loss1 + self.loss2
		self.loss.backward()

	def optimize(self):
		self.forward()
		self.netD = torch.tensor(self.netD, requires_grad=True)
		self.opD.zero_grad()
		self.backwardD()
		self.opD.step()
		self.netD = torch.tensor(self.netD, requires_grad=False)
		self.opG.zero_grad()
		self.backwardG()
		self.opG.step()


