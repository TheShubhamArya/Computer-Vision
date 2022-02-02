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

# function tat returns the resized image
def get_resized_img(img, new_row, new_col):
    # Create a new array with those dimensions
    new_img = np.zeros([new_row,new_col,img.shape[2]])
    
    # loop through pixel in the new array
    for i in range(new_row):
        for j in range(new_col):
            new_img[i][j] = img[int(i*2)][int(j*2)]
    
    return new_img
    
# resize image with the given factor
def resize_img(img, height):
    h = 1
    img = img
    factor = 1
    while h < height:
        shape = img.shape
        image_resized = get_resized_img(img, shape[0]//2, shape[1]//2)
#        image_resized = transform.resize(img, [shape[0]//2, shape[1]//2,3],anti_aliasing=True)
        factor *= 2
        filename = "img_x"+str(factor)+".png"
        io.imsave(filename, image_resized)
        print("Resized image saved to folder as ",filename)
        img = image_resized
        h += 1
    
if __name__ == '__main__':
    img = io.imread("img.png")
    resize_img(img, 5)
