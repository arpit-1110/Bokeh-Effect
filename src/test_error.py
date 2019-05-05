import torch
import cv2
import sys
from utils import dice_coeff,dice_loss,normalization,denormalize,ab_rel_diff,sq_rel_diff,rms_linear
from data_loader import NYU_Depth_V2, NYU_Depth_V2_v2

def set_requires_grad(nets, requires_grad=False):

    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


modelName = sys.argv[1]
modelType = sys.argv[2]

test_dataset = NYU_Depth_V2('test')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if modelType == 'c' :
    model = torch.load("../CGmodel/"+modelName)
    gen = model.G_XtoY
elif modelType == 'p' :
    model = torch.load("../P2Pmodel/"+modelName)
    gen = model.G
else:
    print("Choose a model type from 'c/p'")
    exit(1)

# p2p_val_loader = torch.utils.data.DataLoader(p2p_dataset[1], batch_size = 4, shuffle=True, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=1)
test_iter = iter(test_dataloader)
dice_val = 0
rel_val = 0
sq_val = 0
rms_val = 0
gen = gen.to(device)
set_requires_grad(gen,False)
count=0
print(len(test_dataset))
for i in range(int(len(test_dataset)/1)):
    X,Y = test_iter.next()
    X = normalization(X).to(device)
    Y = normalization(Y).to(device)
    gen_Y = gen.forward(X)
    gen_Y = denormalize(gen_Y,flag=1)
    Y = denormalize(Y,flag=1)
    # dice_s = dice_coeff(gen_Y,Y)
    rel_diff = ab_rel_diff(gen_Y,Y)
    sq_rel_differ = sq_rel_diff(gen_Y,Y)
    rms = rms_linear(gen_Y,Y)
    # dice_val += dice_s
    rel_val += rel_diff
    sq_val += sq_rel_differ
    rms_val += rms
    # print("Dice coefficient for batch ", i+1, " : ", dice_s)
    print("Absolute relative difference for batch ", i+1, " : ", rel_diff)
    print("Square relative difference for batch ", i+1, " : ", sq_rel_differ)
    print("RMS loss for batch ", i+1, " : ", rms)
    count= count+1

# print("Average Dice Coefficient: ", dice_val/count)
print("Average absolute relative difference : ", rel_val/count)
print("Averag Square relative difference: ", sq_val/count)
print("Average RMS loss: ", rms_val/count)
