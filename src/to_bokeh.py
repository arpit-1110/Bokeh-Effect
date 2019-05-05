import cv2
import numpy as np
import torch
import sys
import torch.nn.functional as F


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

imageName = sys.argv[1]
depthMap = sys.argv[2]
def adjust_gamma(img, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(img, table)

def disc_blur(img, r):
    # image = torch.from_numpy(img).float()
    # print(type(img))
    # image = img.reshape((256, 256, 3))
    image=img
    image = image
    image = image
    # print(img[0,:,:].shape)
    final_img = np.zeros(image.shape, dtype='uint8')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # if i > 0 and j > 0 and i < 255 and j < 255:
            #   final_img[i, j] = (image[i-1, j-1] + image[i-1, j] + image[i-1, j+1] + image[i, j+1] +
            #               image[i+1, j+1] + image[i+1, j] + image[i+1, j-1] + image[i, j-1]
            #               + image[i, j])//9
            # else:
            #   final_img[i, j] = image[i, j]
            count = 0
            all_pixel = np.zeros(image[i, j].shape)
            for a in range(-r,r):
                for b in range(-r,r):
                    if a**2 + b**2 < r**2 and a+i > 0 and b+j > 0 and a+i < 255 and b+j < 255:
                        count += 1
                        all_pixel += image[a+i, b+j]
            all_pixel = all_pixel/count
            final_img[i, j] = all_pixel
    return final_img


from utils import get_faces

def in_face(i, j, faces):
    for face in faces:
        # print(j > face[0])
        if j > face[0] and i > face[1] and j < face[0] + face[2] and i < face[1] + face[3]:
            return True
    return False

def to_bokeh(img, depth, gamma, thresh, down):
    image = cv2.imread(img)
    depth_mask = np.array(cv2.imread(depth))
    # print(image.shape)
    gamma_corrected_img = adjust_gamma(image, gamma)
    bokeh = disc_blur(gamma_corrected_img, 5)
    # print(bokeh.shape)
    bokeh = np.array(adjust_gamma(bokeh, 1/gamma))
    blurImg = disc_blur(image, 5)
    bokeh = np.maximum(bokeh, blurImg)
    faces = get_faces(img)
    print(faces)
    image = np.array(image)
    final_img = np.zeros(image.shape)
    print(image.shape)
    max_depth = np.max(depth_mask)
    print(max_depth)
    if len(faces) == 0:
        thresh = 2
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if in_face(i, j, faces):
                depth_mask[i, j] = depth_mask[i, j]//down
            if np.mean(depth_mask[i, j]) > max_depth//thresh:
                final_img[i, j] = bokeh[i, j]
            else :
                final_img[i, j] = image[i, j]
    cv2.imwrite('bokeh.jpg', final_img)


to_bokeh(imageName, depthMap, 20.0, 2.5, 2.5)
