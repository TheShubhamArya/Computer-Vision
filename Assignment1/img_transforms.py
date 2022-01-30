"""
Shubham Arya 1001650536
CSE 4310 Computer Vision Spring 2022 UTA
Assignment 1 - Image transformations
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
def resize_img(img, factor):
    shape = img.shape
    image_resized = transform.resize(img, [int(shape[0]*factor), int(shape[1]*factor),3],anti_aliasing=True)
    filename = "imgx"+str(factor)+".jpeg"
    io.imsave(filename, image_resized)
    print("Resized image saved to folder as ",filename)
    
def rgb_to_hsv(img,h,s,v):
    img_hsv = color.rgb2hsv(img)
    img_hsv[:, :, 0] += random.uniform(0,h)
    img_hsv[:, :, 1] += random.uniform(0,s)
    img_hsv[:, :, 2] += random.uniform(0,v)
    return img_hsv

def hsv_to_rgb(img):
    img_rgb = color.hsv2rgb(img)
    return img_rgb

def color_jitter(img, h, s, v):
    if h < 0 or h > 360:
        print("Value of hue should be between 0 and 360")
    elif s < 0 or s > 1:
        print("Value of saturation should be betweeen 0 and 1")
    elif v < 0 or v > 1:
        print("Value of value inputs should be betweeen 0 and 1")
    else:
        hsv_img = rgb_to_hsv(img,h,s,v)
        rgb_img = hsv_to_rgb(hsv_img)
        io.imsave('img_color_jitter.jpeg', rgb_img)
        print("New img_color_jitter.jpeg added to folder.")

if __name__ == '__main__':
    img = io.imread("img.jpeg")
#    random_crop(img, 183)
#    extract_patch(img,2)
#    resize_img(img,0.5)
    color_jitter(img, 0.4, 0.5, 0.3)
    
#    if len(sys.argv) != 5:
#        print("Error: Incorrect format!\nThe run command is: python3 change_hsv.py [filename] [hue value] [saturation value] [value modification]")
#    else:
#        args = sys.argv
#        color_spaces(args[1], float(args[2]), float(args[3]), float(args[4]))
