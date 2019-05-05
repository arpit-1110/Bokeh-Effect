import torch
import torch.nn as nn
import torch.optim as optim
from cnn_utils import CycleGenerator
from cnn_utils import RensetGenerator, PatchDiscriminator
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class cycleGan():

    # def __init__(self,  g_conv_dim=64, d_conv_dim=64,res_blocks=4,lr=0.001, beta1=0.5, beta2=0.999):
    def __init__(self,opt):
        super(cycleGan, self).__init__()
        self.opt = opt
        self.G_XtoY = RensetGenerator(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, not opt.no_dropout, opt.n_blocks, opt.padding_type).to(device)
        # self.G_YtoX = CycleGenerator(d_conv_dim, res_blocks)
        self.G_YtoX = RensetGenerator(opt.output_nc, opt.input_nc, opt.ngf, opt.norm, not opt.no_dropout, opt.n_blocks, opt.padding_type).to(device)
        # self.D_X = CycleDiscriminator(d_conv_dim)
        self.D_X = PatchDiscriminator(opt.input_nc, opt.ndf, opt.n_layers_D, opt.norm).to(device)
        # self.D_Y = CycleDiscriminator(d_conv_dim)
        self.D_Y = PatchDiscriminator(opt.output_nc, opt.ndf, opt.n_layers_D, opt.norm).to(device)
        # self.G_XtoY.apply(init_weights)
        # self.G_YtoX.apply(init_weights)
        # self.D_X.apply(init_weights)
        # self.D_Y.apply(init_weights)
        print(self.G_XtoY)
        print("Parameters: " ,len(list(self.G_XtoY.parameters())))
        print(self.G_YtoX)
        print("Parameters: " ,len(list(self.G_YtoX.parameters())))
        print(self.D_X)
        print("Parameters: " ,len(list(self.D_X.parameters())))
        print(self.D_Y)
        print("Parameters: " ,len(list(self.D_Y.parameters())))
        self.fake_X_pool = ImagePool(opt.pool_size)
        self.fake_Y_pool = ImagePool(opt.pool_size)
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()  #For implementing Identity Loss
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_XtoY.parameters(), self.G_YtoX.parameters()), lr=opt.lr, betas=[opt.beta1, 0.999])
        self.optimizer_DX = torch.optim.Adam(self.D_X.parameters(), lr=opt.lr, betas=[opt.beta1, 0.999])
        self.optimizer_DY = torch.optim.Adam(self.D_Y.parameters(), lr=opt.lr, betas=[opt.beta1, 0.999])

        # self.genParams = list(self.G_XtoY.parameters()) + list(self.G_YtoX.parameters())
        # self.g_optim = optim.Adam(self.genParams,lr,[beta1,beta2])
        # self.dx_optim = optim.Adam(self.D_X.parameters(),lr,[beta1,beta2])
        # self.dy_optim = optim.Adam(self.D_Y.parameters(),lr,[beta1,beta2])

    def get_input(self, inputX, inputY):
        self.inputX = inputX
        self.inputY = inputY

    def forward(self):
        self.fake_X = self.G_YtoX(self.inputY).to(device)
        self.rec_Y= self.G_XtoY(self.fake_X).to(device)
        self.fake_Y = self.G_XtoY(self.inputX).to(device)
        self.rec_X = self.G_YtoX(self.fake_Y).to(device)

    def backward_D_basic(self,netD,real,fake):
        pred_real = netD(real)
        loss_D_real = real_mse_loss(pred_real)        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = fake_mse_loss(pred_fake)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_X(self):
        """Calculate GAN loss for discriminator D_A"""
        if self.opt.pool==True:
            fake_X = self.fake_X_pool.query(self.fake_X)
            self.loss_D_X = self.backward_D_basic(self.D_X, self.inputX, fake_X)
        else:
            self.loss_D_X = self.backward_D_basic(self.D_X, self.inputX, self.fake_X)

    def backward_D_Y(self):
        """Calculate GAN loss for discriminator D_A"""
        if self.opt.pool==True:
            fake_Y = self.fake_Y_pool.query(self.fake_Y)
            self.loss_D_Y = self.backward_D_basic(self.D_Y, self.inputY, fake_Y)
        else:
            self.loss_D_Y = self.backward_D_basic(self.D_Y, self.inputY, self.fake_Y)

    def backward_G(self):
        # Not implemented identity loss

        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        self.loss_G_X = real_mse_loss(self.D_X(self.fake_X))
        # GAN loss D_B(G_B(B))
        self.loss_G_Y = real_mse_loss(self.D_Y(self.fake_Y))
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_X = self.criterionCycle(self.rec_X, self.inputX) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_Y = self.criterionCycle(self.rec_Y, self.inputY) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_X+ self.loss_G_Y + self.loss_cycle_X + self.loss_cycle_Y
        self.loss_G.backward()

    def change_lr(self, new_lr):
        self.opt.lr = new_lr
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_XtoY.parameters(), self.G_YtoX.parameters()), lr=self.opt.lr, betas=[self.opt.beta1, 0.999])
        self.optimizer_DX = torch.optim.Adam(self.D_X.parameters(), lr=self.opt.lr, betas=[self.opt.beta1, 0.999])
        self.optimizer_DY = torch.optim.Adam(self.D_Y.parameters(), lr=self.opt.lr, betas=[self.opt.beta1, 0.999])





    # def trainDX(self, inputX, inputY):
    #     self.dx_optim.zero_grad()
    #     out_X = self.D_X(inputX)
    #     real_loss = real_mse_loss(out_X)

    #     fake_X = self.G_YtoX(inputY)
    #     out_X_fake = self.D_X(fake_X)
    #     fake_loss = fake_mse_loss(out_X_fake)

    #     loss = fake_loss + real_loss
    #     self.dx_loss = loss
    #     loss = loss.backward()
    #     self.dx_optim.step()

    # def trainDY(self, inputX, inputY):
    #     self.dy_optim.zero_grad()
    #     out_Y = self.D_Y(inputY)
    #     real_loss = real_mse_loss(out_Y)

    #     fake_Y = self.G_XtoY(inputX)
    #     out_Y_fake = self.D_Y(fake_Y)
    #     fake_loss = fake_mse_loss(out_Y_fake)

    #     loss = fake_loss + real_loss
    #     self.dy_loss = loss
    #     loss = loss.backward()
    #     self.dy_optim.step()

    # def trainGen(self, inputX, inputY):
    #     self.g_optim.zero_grad()

    #     fake_X = self.G_YtoX(inputY)
    #     out_X = self.D_X(fake_X)
    #     g_YtoX_loss = real_mse_loss(out_X)

    #     reconstructed_Y = self.G_XtoY(fake_X)
    #     reconstructed_y_loss = cycle_consistency_loss(inputY, reconstructed_Y, lambda_weight=10)

    #     fake_Y = self.G_XtoY(inputX)
    #     out_Y = self.D_Y(fake_Y)
    #     g_XtoY_loss = real_mse_loss(out_Y)

    #     reconstructed_X = self.G_YtoX(fake_Y)
    #     reconstructed_x_loss = cycle_consistency_loss(inputX, reconstructed_X, lambda_weight=10)

    #     g_total_loss = g_YtoX_loss + g_XtoY_loss + reconstructed_y_loss + reconstructed_x_loss
    #     self.gen_loss = g_total_loss
    #     g_total_loss.backward()
    #     self.g_optim.step()

    def set_requires_grad(self, nets, requires_grad=False):

        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def optimize(self):
        # print(self.inputX.size())
        # print(self.inputY.size())
        # a = self.inputX
        # b = self.inputY
        # self.trainDX(a, b)
        # self.trainDY(a, b)
        # self.trainGen(a, b)
        self.set_requires_grad([self.D_X, self.D_Y], False)
        self.forward()
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights


        self.set_requires_grad([self.D_X, self.D_Y], True)
        self.forward()
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()

        # D_A and D_B
        self.optimizer_DX.zero_grad()
        self.optimizer_DY.zero_grad()    # set D_A and D_B's gradients to zero
        self.backward_D_X()      # calculate gradients for D_A
        self.backward_D_Y()      # calculate graidents for D_B
        self.optimizer_DX.step()  # update D_A and D_B's weights
        self.optimizer_DY.step()  # update D_A and D_B's weights
