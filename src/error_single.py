import torch
import cv2
import sys
from utils import dice_coeff,dice_loss,normalization,denormalize,ab_rel_diff,sq_rel_diff,rms_linear
import numpy as np
def set_requires_grad(nets, requires_grad=False):

    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


modelName = sys.argv[1]
modelType = sys.argv[2]

imagePath = sys.argv[3]
depthMap = sys.argv[4]

image = cv2.resize(cv2.imread(imagePath),(256,256), interpolation=cv2.INTER_CUBIC)
depth = cv2.resize(cv2.imread(depthMap),(256,256), interpolation=cv2.INTER_CUBIC)
image = torch.from_numpy(np.array(image).reshape(1,3,256,256)).float()
depth = torch.from_numpy(np.array(depth).reshape(1,3,256,256)).float()


if modelType == 'c' :
    model = torch.load("../CGmodel/"+modelName)
    gen = model.G_XtoY
elif modelType == 'p' :
    model = torch.load("../P2Pmodel/"+modelName)
    gen = model.G
else:
    print("Choose a model type from 'c/p'")
    exit(1)

set_requires_grad(gen,False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image = normalization(image).to(device)
pred_depth = gen.to(device).forward(image)
depth = normalization(depth).to(device)
cv2.imwrite("testDepth.jpg", np.array(denormalize(depth).cpu().detach()).reshape(256,256,3))
pred_depth = denormalize(pred_depth,flag=1)
depth = denormalize(depth,flag=1)
# dice=dice_coeff(pred_depth,depth)
rel_dif = ab_rel_diff(pred_depth,depth)
sq_rel_dif = sq_rel_diff(pred_depth,depth)
rms = rms_linear(pred_depth,depth)
# print("Dice Coefficient is : ", dice)
print("Absolute Relative Difference is : ", rel_dif)
print("Square Relative Difference is : ", sq_rel_dif)
print("RMS Difference is : ", rms)
