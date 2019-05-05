import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
import sys

if sys.argv[1] == 'c':
    modelNumber = sys.argv[2]
    loss_dir = os.path.join("..", "CGloss")

    loss_DX_f = loss_dir + "/lossD_X" + modelNumber +".pk"
    loss_DY_f = loss_dir + "/lossd_Y" + modelNumber +".pk"
    loss_G_f = loss_dir + "/lossG" + modelNumber +".pk"

    loss_DX = pickle.load(open(loss_DX_f,'rb'))
    loss_DY = pickle.load(open(loss_DY_f,'rb'))
    loss_G = pickle.load(open(loss_G_f,'rb'))

    # print(loss_DX)
    plt.plot([x+1 for x in range(len(loss_DX))], loss_DX)
    plt.savefig('loss_Dx.png')
    plt.plot([x+1 for x in range(len(loss_DY))], loss_DY)
    plt.savefig('loss_Dy.png')
    plt.plot([x+1 for x in range(len(loss_G))], loss_G)
    plt.gca().legend(('Discriminator_X loss','Discriminator_Y loss', 'Generator loss'))
    plt.savefig('loss_G.png')

if sys.argv[1] == 'p':
    modelNumber = sys.argv[2]
    loss_dir = os.path.join("..", "P2Ploss")

    loss_D_f = loss_dir + "/lossD" + modelNumber +".pk"
    loss_G_f = loss_dir + "/lossG" + modelNumber +".pk"

    loss_D = pickle.load(open(loss_D_f,'rb'))
    loss_G = pickle.load(open(loss_G_f,'rb'))

    # print(loss_DX)
    plt.plot([x+1 for x in range(len(loss_D))], loss_D)
    plt.savefig('loss_Dx.png')
    plt.plot([x+1 for x in range(len(loss_G))], loss_G)
    plt.gca().legend(('Discriminator loss', 'Generator loss'))
    plt.savefig('loss_G.png')
