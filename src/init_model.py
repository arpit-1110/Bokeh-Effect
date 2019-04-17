import torch
import torch.nn as nn

def init_weights(m):
    if type(m) in [nn.Conv2d, nn.ConvTranspose2d]:
        torch.nn.init.xavier_uniform_(m.weight)
