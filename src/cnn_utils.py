import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
	torch.set_default_tensor_type('torch.FloatTensor')

###### Pix2Pix Utilities ##########################################################

class downConv(nn.Module):
	def __init__(self, in_layer, out_layer, take_norm=True):
		super(downConv, self).__init__()
		self.conv = nn.Conv2d(in_layer, out_layer, kernel_size=4, stride=2, padding=1)
		self.act = nn.LeakyReLU(0.2, True)
		self.norm = nn.BatchNorm2d(out_layer)
		self.take_norm = take_norm

	def forward(self, x):
		x = self.conv(x)
		if self.take_norm:
			return self.act(self.norm(x))
		else :
			return self.act(x)

class upConv(nn.Module):
	def __init__(self, in_layer, out_layer):
		super(upConv, self).__init__()
		self.convt = nn.ConvTranspose2d(in_layer, out_layer, kernel_size=4, stride=2, padding=1)
		self.act = nn.ReLU(True)
		self.norm = nn.BatchNorm2d(out_layer)

	def forward(self, x):
		x = self.act(self.norm(self.convt(x)))
		return x

class Generator(nn.Module):
	def __init__(self, n_downsample=3, n_channels=3):
		super(Generator, self).__init__()
		model = [downConv(n_channels, 64, take_norm=False)]
		model += [downConv(64, 128)]
		model += [downConv(128, 256)]
		model += [downConv(256, 512)]
		for i in range(n_downsample):
			model += [downConv(512, 512)]

		for i in range(n_downsample):
			model += [upConv(512, 512)]

		model += [upConv(512, 256)]
		model += [upConv(256, 128)]
		model += [upConv(128, 64)]
		model += [nn.ConvTranspose2d(64, n_channels, kernel_size=4, stride=2, padding=1)]
		model += [nn.ReLU(True)]
		model += [nn.Tanh()]

		self.model = nn.Sequential(*model)

	def forward(self, x):
		return self.model(x)

class Discriminator(nn.Module):
	def __init__(self, n_channels=6):
		super(Discriminator, self).__init__()
		model = [downConv(n_channels, 64, take_norm=False)]
		model += [downConv(64, 128)]
		model += [downConv(128, 256)]
		model += [nn.Conv2d(256, 512, kernel_size=2, stride=1)]
		model += [nn.BatchNorm2d(512)]
		model += [nn.LeakyReLU(0.2, True)]
		model += [nn.Conv2d(512, 1, kernel_size=2, stride=1)]
		model += [nn.Sigmoid()]

		self.model = nn.Sequential(*model)

	def forward(self, inp, un):
		return self.model(torch.cat((inp, un), 1))




# if __name__ == '__main__':
# 	inp = torch.randn(1, 3, 256, 256)
# 	un = torch.randn(1, 3, 256, 256)
# 	model = Discriminator()
# 	# model2 = downConv(3, 64)
# 	# print(model)
# 	print(model(inp, un))

#######################################################################################

################CycleGAN Utilities#####################################################

def c_downConv(in_layer,out_layer,kernel_size,stride=2,padding=1,batch_norm=True):
	layers=[]
	layers+=[nn.Conv2d(in_layer,out_layer,kernel_size,stride,padding,bias=False)]
	if batch_norm:
		layers+=[nn.BatchNorm2d(out_layer)]


	return nn.Sequential(*layers)

def c_upConv(in_layer,out_layer, kernel_size, stride=2, padding=1, batch_norm=True):
	layers = []
	layers+=[nn.ConvTranspose2d(in_layer,out_layer,kernel_size,stride,padding,bias=False)]
	if batch_norm:
		layers+=[nn.BatchNorm2d(out_layer)]

	return nn.Sequential(*layers)

class ResidualBlock(nn.Module):

	def __init__(self, conv_dim):
		super(ResidualBlock,self).__init__()

		self.conv_layer1 = c_downConv(conv_dim,conv_dim,3,1,1,True)
		self.conv_layer2 = c_downConv(conv_dim,conv_dim,3,1,1,True)

	def forward(self,x):
		out = F.relu(self.conv_layer1(x))
		out = x + self.conv_layer2(out)
		return out

class CycleDiscriminator(nn.Module):
	def __init__(self,conv_dim=64):
		self.conv1 = c_downConv(3,conv_dim,4,batch_norm=False)
		self.conv2 = c_downConv(conv_dim,conv_dim*2,4)
		self.conv3 = c_downConv(conv_dim*2,conv_dim*4,4)
		self.conv4 = c_downConv(conv_dim*4,conv_dim*8,4)

		self.conv5 = c_downConv(conv_dim*8,1,4,stride=1,batch_norm=False)


	def forward(self,x):
		out = F.relu(self.conv1(x))
		out = F.relu(self.conv2(out))
		out = F.relu(self.conv3(out))
		out = F.relu(self.conv4(out))

		out = self.conv5(out)
		out = nn.Sigmoid(out)
		return out


class CycleGenerator(nn.Module):
	def __init__(self,conv_dim=64, res_blocks=6):
		super(CycleGenerator,self).__init__()
		self.conv1 = c_downConv(3,conv_dim,4)
		self.conv2 = c_downConv(conv_dim,conv_dim*2,4)
		self.conv3 = c_downConv(conv_dim*2,conv_dim*4,4)

		res_layers=[]
		for i in range(res_blocks):
			res_layers+=ResidualBlock(conv_dim*4)

		self.res_blocks=nn.Sequential(*res_layers)

		self.deconv1 = c_upConv(conv_dim*4,conv_dim*2,4)
		self.deconv2 = c_upConv(conv_dim*2,conv_dim,4)
		self.deconv3 = c_upConv(conv_dim,3,4,batch_norm=False)

	def forward(self,x):
		out = F.relu(self.conv1(x))
		out = F.relu(self.conv2(out))
		out = F.relu(self.conv3(out))

		out = self.res_blocks(out)

		out = F.relu(self.deconv1(out))
		out = F.relu(self.deconv2(out))
		out = F.tanh(self.deconv3(out))

		return out






