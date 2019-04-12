import torch
import torch.nn as nn
import torch.optim as optim
from cnn_utils import CycleGenerator
from cnn_utils import CycleDiscriminator
from loss_utils import real_mse_loss, fake_mse_loss, cycle_consistency_loss


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

class cycleGan():

    def __init__(self,  g_conv_dim=64, d_conv_dim=64,res_blocks=4,lr=0.001, beta1=0.5, beta2=0.999):
        super(cycleGan, self).__init__()
        self.G_XtoY = CycleGenerator(g_conv_dim, res_blocks)
        self.G_YtoX = CycleGenerator(d_conv_dim, res_blocks)
        self.D_X = CycleDiscriminator(d_conv_dim)
        self.D_Y = CycleDiscriminator(d_conv_dim)
        self.genParams = list(self.G_XtoY.parameters()) + list(self.G_YtoX.parameters())
        self.g_optim = optim.Adam(self.genParams,lr,[beta1,beta2])
        self.dx_optim = optim.Adam(self.D_X.parameters(),lr,[beta1,beta2])
        self.dy_optim = optim.Adam(self.D_Y.parameters(),lr,[beta1,beta2])

    def get_input(self, inputX, inputY):
        self.inputX = inputX
        self.inputY = inputY


    def trainDX(self, inputX, inputY):
        self.dx_optim.zero_grad()
        out_X = self.D_X(inputX)
        real_loss = real_mse_loss(out_X)

        fake_X = self.G_YtoX(inputY)
        out_X_fake = self.D_X(fake_X)
        fake_loss = fake_mse_loss(out_X_fake)

        loss = fake_loss + real_loss
        self.dx_loss = loss
        loss = loss.backward()
        self.dx_optim.step()

    def trainDY(self, inputX, inputY):
        self.dy_optim.zero_grad()
        out_Y = self.D_Y(inputY)
        real_loss = real_mse_loss(out_Y)

        fake_Y = self.G_XtoY(inputX)
        out_Y_fake = self.D_Y(fake_Y)
        fake_loss = fake_mse_loss(out_Y_fake)

        loss = fake_loss + real_loss
        self.dy_loss = loss
        loss = loss.backward()
        self.dy_optim.step()

    def trainGen(self, inputX, inputY):
        self.g_optim.zero_grad()

        fake_X = self.G_YtoX(inputY)
        out_X = self.D_X(fake_X)
        g_YtoX_loss = real_mse_loss(out_X)

        reconstructed_Y = self.G_XtoY(fake_X)
        reconstructed_y_loss = cycle_consistency_loss(inputY, reconstructed_Y, lambda_weight=10)

        fake_Y = self.G_XtoY(inputX)
        out_Y = self.D_Y(fake_Y)
        g_XtoY_loss = real_mse_loss(out_Y)

        reconstructed_X = self.G_YtoX(fake_Y)
        reconstructed_x_loss = cycle_consistency_loss(inputX, reconstructed_X, lambda_weight=10)

        g_total_loss = g_YtoX_loss + g_XtoY_loss + reconstructed_y_loss + reconstructed_x_loss
        self.gen_loss = g_total_loss
        g_total_loss.backward()
        self.g_optim.step()

    def optimize(self):
        # print(self.inputX.size())
        # print(self.inputY.size())
        a = self.inputX
        b = self.inputY
        self.trainDX(a, b)
        self.trainDY(a, b)
        self.trainGen(a, b)






