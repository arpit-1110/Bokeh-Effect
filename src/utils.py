import torch
import sys
import math
import random


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

def scale(X):
    return 2*X-1

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

class cgOptimizer():
    def __init__(self, input_nc=None, output_nc=None, ngf=None, norm=None, no_dropout=None, n_blocks=None,
                 padding_type=None, ndf=None, n_layers_D = None, pool_size = None, lr = None, beta1 = None, lambda_A = None, lambda_B = None, pool=None):
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.norm = norm
        self.no_dropout = no_dropout
        self.n_blocks = n_blocks
        self.padding_type = padding_type
        self.ndf = ndf
        self.n_layers_D = n_layers_D
        self.pool_size = pool_size
        self.lr=lr
        self.beta1 = beta1
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.pool = pool

class p2pOptimizer():
    def __init__(self, input_nc=None, output_nc=None, num_downs=None, ngf=None, norm_layer=None, use_dropout=None, ndf=None, n_layers_D=None, lr=None, beta1=None, lambda_L1=None, n_blocks=None, padding_type=None):
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.num_downs=num_downs
        self.norm_layer=norm_layer
        self.use_dropout=use_dropout
        self.ndf=ndf
        self.n_layers_D=n_layers_D
        self.lr=lr
        self.beta1=beta1
        self.lambda_L1=lambda_L1
        self.n_blocks=n_blocks
        self.padding_type=padding_type


class ImagePool():

    def __init__(self, pool_size):

        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):

        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images
