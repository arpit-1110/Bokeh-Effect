import torch
import sys
import cv2
from utils import normalization,denormalize
import numpy as np

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

bestModel = sys.argv[1]
testImage = sys.argv[2]
modelType = sys.argv[3]


testImage = cv2.resize(cv2.imread(testImage),(256,256), interpolation=cv2.INTER_CUBIC)
cv2.imwrite( './testInput.jpeg',np.array(testImage).reshape(256, 256, 3))
testImage = torch.from_numpy(np.array(testImage).reshape(1,3,256,256)).float()
testImage = normalization(testImage).to(device)
print(testImage)
if modelType == 'p':
    model = torch.load("../P2Pmodel/" + bestModel)
    output = model.G.forward(testImage)
else:
    model = torch.load("../CGmodel/" + bestModel)
    output = model.G_XtoY.forward(testImage)

cv2.imwrite( './testOutput.jpeg',np.array(denormalize(output).cpu().detach()).reshape(256, 256, 3))
