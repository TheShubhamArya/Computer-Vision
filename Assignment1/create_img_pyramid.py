"""
Shubham Arya 1001650536
CSE 4310 Computer Vision Spring 2022 UTA
Assignment 1 - Image pyramids
"""
import numpy as np
import random
import sys
import skimage.io as io
import skimage.color as color
from skimage import transform

# generates random square crops of an image with the image and size of the cropped image given
def random_crop(img, size):
    img_size = img.shape
    min_dimension = min(img_size[0], img_size[1])
    if size <= min_dimension:
        x = random.randint(0, img_size[0] - size)
        y = random.randint(0, img_size[1] - size)
        x2 = x + size
        y2 = y + size
        img_crop = img[x:x2, y:y2]
        return img_crop
#        io.imsave('crop_image.jpeg', img_crop)
#        print("Cropped image has been added to the folder.")
    else:
        print("crop size is bigger than the image")
    
# Generates n^2 patches of an image with the size n.
def extract_patch(img, num_patches):
    size = num_patches
    img = img
    img_size = img.shape
    if img_size[0] != img_size[1]:
        min_dimension = min(img_size[0], img_size[1])
        img = random_crop(img,min_dimension)
    H,W = img.shape[0], img.shape[1]
    rows, cols = size,size
    patchSize = img.shape[0]//size
    for i in range(rows):
        for j in range(cols):
            new_image = img[i*patchSize:patchSize*(i+1), j*patchSize:(j+1)*patchSize]
            filename = "patch"+str(i)+str(j)+".jpeg"
            io.imsave(filename, new_image)
    print("Image patches added to the folder")
    
# resize image with the given factor
def resize_img(img, height):
    h = 1
    img = img
    factor = 1
    while h < height:
        shape = img.shape
        image_resized = transform.resize(img, [shape[0]//2, shape[1]//2,3],anti_aliasing=True)
        factor = factor*2
        filename = "img_x"+str(factor)+".jpeg"
        io.imsave(filename, image_resized)
        print("Resized image saved to folder as ",filename)
        img = image_resized
        h += 1
    
if __name__ == '__main__':
    img = io.imread("img.jpeg")
    resize_img(img, 4)
