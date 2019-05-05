import torch
import torch.nn as nn
import sys
from pix2pix import Pix2Pix
from cycleGAN import cycleGan
from utils import Optimizer, cgOptimizer, p2pOptimizer
from torch.optim import lr_scheduler
import time
import copy
from torch.autograd import Variable
from utils import dice_coeff, normalization, denormalize
from data_loader import NYU_Depth_V2, NYU_Depth_V2_v2
import cv2
import numpy as np
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pickle
import os

modelName = sys.argv[1]
modelType = sys.argv[2]
testImage = sys.argv[3]

os.system("python3 test_model.py "+ modelName + " "+ testImage + " " + modelType)
os.system("python3 to_bokeh.py testInput.jpeg testOutput.jpeg")
os.system("rm -rf testInput.jpeg testOutput.jpeg")
