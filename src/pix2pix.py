import torch
import torch.nn as nn
import torch.optim as optim
from cnn_utils import CycleGenerator
from cnn_utils import RensetGenerator, PatchDiscriminator, UnetGenerator
from utils import ImagePool
from cnn_utils import CycleDiscriminator
import itertools
from loss_utils import real_mse_loss, fake_mse_loss, cycle_consistency_loss
from init_model import init_weights


# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
# else:
#     torch.set_default_tensor_type('torch.FloatTensor')

torch.set_default_tensor_type('torch.FloatTensor')

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class Pix2Pix():
    #def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
    # def __init__(self,  g_conv_dim=64, d_conv_dim=64,res_blocks=4,lr=0.001, beta1=0.5, beta2=0.999):
    def __init__(self,opt):
        super(Pix2Pix, self).__init__()
        self.opt = opt
        self.G = UnetGenerator(opt.input_nc, opt.output_nc, opt.num_downs, opt.ngf, opt.norm_layer, opt.use_dropout).to(device)
        # self.G = RensetGenerator(opt.input_nc, opt.output_nc, opt.ngf, opt.norm_layer, opt.use_dropout, opt.n_blocks, opt.padding_type).to(device)
        self.D = PatchDiscriminator(self.opt.input_nc + self.opt.output_nc, self.opt.ndf, self.opt.n_layers_D, self.opt.norm_layer).to(device)
        print(self.G)
        print("Parameters: " ,len(list(self.G.parameters())))
        print(self.D)
        print("Parameters: " ,len(list(self.D.parameters())))
        self.L1Loss = torch.nn.L1Loss()
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.opt.lr, betas=[self.opt.beta1, 0.999])
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.opt.lr, betas=[self.opt.beta1, 0.999])

    def get_input(self, inputX, inputY):
        self.inputX = inputX.to(device)
        self.inputY = inputY.to(device)

    def forward(self):
        self.fake_Y = self.G(self.inputX).to(device)

    def backward_D(self):
        fake_XY = torch.cat((self.inputX, self.fake_Y), 1)
        pred_fake = self.D(fake_XY.detach())
        self.loss_D_fake = fake_mse_loss(pred_fake)
        real_XY = torch.cat((self.inputX, self.inputY), 1)
        pred_real = self.D(real_XY)
        self.loss_D_real = real_mse_loss(pred_real)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):

        fake_XY = torch.cat((self.inputX, self.fake_Y), 1)
        pred_fake = self.D(fake_XY)
        self.loss_G_GAN = real_mse_loss(pred_fake)
        self.loss_G_L1 = self.L1Loss(self.inputY, self.fake_Y) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def change_lr(self, new_lr):
        self.opt.lr = new_lr
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.opt.lr, betas=[self.opt.beta1, 0.999])
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.opt.lr, betas=[self.opt.beta1, 0.999])


    def set_requires_grad(self, nets, requires_grad=False):

        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def optimize(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.D, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.D, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weight

        self.forward()
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()
        self.set_requires_grad(self.D, True)
