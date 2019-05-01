import torch
import torch.nn as nn
import functools
import torch.nn.functional as F

# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
# else:
#     torch.set_default_tensor_type('torch.FloatTensor')

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
        super(CycleDiscriminator,self).__init__()
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
        out = F.sigmoid(out)
        # out = nn.Sigmoid(out)
        return out


class CycleGenerator(nn.Module):
    def __init__(self,conv_dim=64, res_blocks=6):
        super(CycleGenerator,self).__init__()
        self.conv1 = c_downConv(3,conv_dim,4)
        self.conv2 = c_downConv(conv_dim,conv_dim*2,4)
        self.conv3 = c_downConv(conv_dim*2,conv_dim*4,4)

        res_layers=[]
        for i in range(res_blocks):
            res_layers.append(ResidualBlock(conv_dim=conv_dim*4))

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


####################################Network Utilities###############################################

def get_norm_layer(norm_type="batch"):
    if(norm_type == "batch"):
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif (norm_type == "instance"):
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        norm_layer = None

    return norm_layer


class RensetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        super(RensetGenerator,self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func==nn.InstanceNorm2d
        else:
            use_bias = norm_layer==nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),nn.ReLU(True)]

        n_downsampling = 2
        init_filter = ngf

        for i in range(n_downsampling):
            model += [nn.Conv2d(init_filter, init_filter*2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(init_filter*2),
                      nn.ReLU(True)]

            init_filter = init_filter*2

        for i in range(n_blocks):
            model += [ResnetBlock(init_filter, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            model += [nn.ConvTranspose2d(init_filter, init_filter//2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                      norm_layer(init_filter//2),
                      nn.ReLU(True)]
            init_filter = init_filter//2

        model+= [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self,input):
        return self.model(input)




class ResnetBlock(nn.Module):

    def __init__(self,dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock,self).__init__()
        model = []
        p = 0
        if padding_type == 'reflect':
            model += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            model += [nn.ReplicationPad2d(1)]
        else:
            p=1

        model += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            model+= nn.Dropout(0.5)

        p = 0
        if padding_type == 'reflect':
            model += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            model += [nn.ReplicationPad2d(1)]
        else:
            p=1

        model += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        self.model = nn.Sequential(*model)

    def forward(self,x):
        out = x + self.model(x)
        return out


class PatchDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):

        super(PatchDiscriminator,self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func!=nn.BatchNorm2d
        else:
            use_bias = norm_layer!=nn.BatchNorm2d

        model = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        init_filter = ndf
        for i in range(1,n_layers):
            model+=[nn.Conv2d(init_filter, init_filter*2, kernel_size=4, stride=2, padding=1, bias=use_bias),
                    norm_layer(init_filter*2),
                    nn.LeakyReLU(0.2, True)
                    ]
            init_filter = init_filter*2

            ## Will have to condition on init_filter if numLayers > 3

        model+=[nn.Conv2d(init_filter, init_filter*2, kernel_size=4, stride=1, padding=1, bias=use_bias),
                norm_layer(init_filter*2),
                nn.LeakyReLU(0.2, True)
                ]

        init_filter = init_filter*2

        model += [nn.Conv2d(init_filter, 1, kernel_size=4, stride=1, padding=1)]  # output 1 channel prediction map
        self.model = nn.Sequential(*model)


    def forward(self,x):
        return self.model(x)


##### For Pix 2 Pix ######

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
