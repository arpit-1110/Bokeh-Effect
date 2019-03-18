import os, shutil
from io import BytesIO
import tarfile, tempfile
from six.moves import urllib
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2


def generate_blur_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian_blurred = cv2.GaussianBlur(img ,(51,51),0)
    median_blurred = cv2.medianBlur(img,15)
    cv2.imwrite('temp/gaussian_blurred.jpg', gaussian_blurred)
    cv2.imwrite('temp/median_blurred.jpg', median_blurred)

# def generate_portrait_mode(real_image_path, seg_image_path):
# 	seg_image = Image.open(seg_image_path)
# 	real_image = Image.open(real_image_path) #Can be many different formats.
# 	seg_image = seg_image.resize((real_image.size[0],real_image.size[1]),Image.ANTIALIAS)
# 	# seg_image.save("temp/new_seg_image", quality = 100) #saved resized image
# 	# seg_image = Image.open("temp/new_seg_image.jpg")

# 	gaussian_blurred = cv2.GaussianBlur(real_image, (51,51),0)

# 	black = 0,0,0
# 	pix = seg_image.load()
# 	pix_blurred = gaussian_blurred.load()
# 	pix_true = real_image.load()
# 	length = real_image.size[0]
# 	breadth = real_image.size[1]
# 	for i in range(1,length):
# 		for j in range(1,breadth):
# 			if(pix[i,j] != black and pix[i,j][2] > 100 and pix[i,j][2] < 140):
# 				pix_blurred[i, j] = pix_true[i, j]
# 	cv2.imwrite('final_image.jpg', gaussian_blurred)


def generate_portrait_mode(orig_path, seg_path):
    ''' try:
        os.remove("/static/output/output.jpg")
    except Exception as e:
        print "Exception." + str(e)
    '''

    # MODEL = load_model()
    orig_image = Image.open(orig_path)

    # createFolder('./temp/')

    generate_blur_image(orig_path)
    # generate_seg_image(path, MODEL)

    seg_image = Image.open(seg_path) #Can be many different formats.
    blur_image = Image.open("temp/gaussian_blurred.jpg")
    seg_image = seg_image.resize((orig_image.size[0],orig_image.size[1]),Image.ANTIALIAS)
    seg_image.save("temp/new_seg_image.jpg", quality = 100) #saved resized image
    seg_image = Image.open("temp/new_seg_image.jpg")

    black = 0,0,0
    pix = seg_image.load()
    pix_blurred = blur_image.load()
    pix_true = orig_image.load()

    length = orig_image.size[0]
    breadth = orig_image.size[1]
    for i in range(1,length):
        for j in range(1,breadth):
            if(pix[i,j] != black and pix[i,j][2] > 100 and pix[i,j][2] < 140):
                pix_blurred[i, j] = pix_true[i, j]


    blur_image.save('final_image.jpg')

real_path = input()
seg_path = input()

generate_portrait_mode(real_path, seg_path)
# generate_blur_image(path)