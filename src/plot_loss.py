import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os

loss_dir = os.path.join("..", "CGloss")

loss_DX_f = loss_dir + "/lossD_X3.pk"
loss_DY_f = loss_dir + "/lossd_Y3.pk"
loss_G_f = loss_dir + "/lossG3.pk"

loss_DX = pickle.load(open(loss_DX_f,'rb'))
loss_DY = pickle.load(open(loss_DY_f,'rb'))
loss_G = pickle.load(open(loss_G_f,'rb'))

# loss_DX = [float(x.cpu().detach()) for x in loss_DX]
# loss_DY = [float(x.cpu().detach()) for x in loss_DY]
# loss_G = [int(x.cpu().detach()) for x in loss_G]

# with open('../CGloss/lossG.pk', 'wb') as f:
#     pickle.dump(loss_G, f)
# with open('../CGloss/lossD_X.pk', 'wb') as f:
#     pickle.dump(loss_DX, f)
# with open('../CGloss/lossd_Y.pk', 'wb') as f:
#     pickle.dump(loss_DY, f)
print(loss_DX)
plt.plot([x+1 for x in range(len(loss_DX))], loss_DX)
plt.savefig('loss_Dx.png')
plt.plot([x+1 for x in range(len(loss_DY))], loss_DY)
plt.savefig('loss_Dy.png')
plt.plot([x+1 for x in range(len(loss_G))], loss_G)
plt.savefig('loss_G.png')
