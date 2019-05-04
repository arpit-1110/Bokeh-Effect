import h5py
import os
import numpy as np
import cv2

matPath = '../data/nyu_depth_v2_labeled.mat'
img_folder = '../data/imgs'
dep_folder = '../data/depths'

if not os.path.exists(img_folder):
    os.makedirs(img_folder)
if not os.path.exists(dep_folder):
    os.makedirs(dep_folder)

f = h5py.File(matPath, 'r')

images = f.get('images').value
depths = f.get('depths').value

img_dim = 256

def save_image_dep(image_id):
    i = image_id
    print ("Processing",i)
    img = images[i]
    depth = depths[i]
    img_=np.empty([img_dim,img_dim,3])
    img_[:,:,0] = cv2.resize(img[2,:,:].T,(img_dim,img_dim))
    img_[:,:,1] = cv2.resize(img[1,:,:].T,(img_dim,img_dim))
    img_[:,:,2] = cv2.resize(img[0,:,:].T,(img_dim,img_dim))
    depth_ = np.empty([img_dim, img_dim, 3])
    depth_[:,:,0] = cv2.resize(depth[:,:].T,(img_dim,img_dim))
    depth_[:,:,1] = cv2.resize(depth[:,:].T,(img_dim,img_dim))
    depth_[:,:,2] = cv2.resize(depth[:,:].T,(img_dim,img_dim))
    img_ = img_  # /255.0
    print(np.amax(depth_))
    depth_ = 255.*cv2.normalize(depth_, 0, 255, cv2.NORM_MINMAX)
    depth_ = np.mean(np.array(depth_),axis=2)

    cv2.imwrite(os.path.join(img_folder,'{}_img.png'.format(i)), img_)
    cv2.imwrite(os.path.join(dep_folder,'{}_depth.png'.format(i)), depth_)


# map(save_image_dep, range(len(f.get('images').value)))
for i in range(len(images)):
    save_image_dep(i)
