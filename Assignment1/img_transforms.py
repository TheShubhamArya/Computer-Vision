"""
Shubham Arya 1001650536
CSE 4310 Computer Vision Spring 2022 UTA
Assignment 1 - Image transformations
"""
import numpy as np
import random
import sys
import math
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
    # Get new dimensions of the new image by multiplying with factor
    new_row = int(img.shape[0]*factor)
    new_col = int(img.shape[1]*factor)
    
    # Create a new array with those dimensions
    new_img = np.zeros([new_row,new_col,img.shape[2]])
    
    # loop through pixel in the new array
    for i in range(new_row):
        for j in range(new_col):
            new_img[i][j] = img[int(i/factor)][int(j/factor)]
    
    filename = "img_x"+str(factor)+".jpeg"
    io.imsave(filename, new_img)
    print("Resized image saved to folder as ",filename)
    
# For testing purposes only
"""
def rgb_to_hsv(img,h,s,v):
    img_hsv = color.rgb2hsv(img)
    img_hsv[:, :, 0] += h
    img_hsv[:, :, 1] += s
    img_hsv[:, :, 2] += v
    return img_hsv

def hsv_to_rgb(img):
    img_rgb = color.hsv2rgb(img)
    return img_rgb
"""
    
# converts rgb to hsv values
def convert_to_hsv(rgb):
    R = rgb[0]
    G = rgb[1]
    B = rgb[2]
    
    R_dash = R / 255
    G_dash = G / 255
    B_dash = B / 255
    
    Cmax = max(R_dash, G_dash, B_dash)
    Cmin = min(R_dash, G_dash, B_dash)
    delta = Cmax - Cmin
    # Hue
    H = 0
    if (delta == 0):
      H = 0
    elif (Cmax == R_dash):
      H = (60 * (((G_dash  - B_dash) / delta) % 6))
    elif (Cmax == G_dash):
      H = (60 * (((B_dash  - R_dash) / delta) + 2))
    elif (Cmax == B_dash):
      H = (60 * (((R_dash  - G_dash) / delta) + 4))
     
    # Saturation
    S = 0
    if (Cmax == 0):
      S = 0
    else:
      S = delta / Cmax
      
    # Value calculation
    V = Cmax
    return H,S,V
    
# converts provided hsv values too rgb values
def convert_to_rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    hBy60 = h / 60.0
    hBy60f = math.floor(hBy60)
    hi = int(hBy60f) % 6
    f = hBy60 - hBy60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q
    r, g, b = float(r*255), float(g*255), float(b*255)
    return r, g, b

# Randomly perturbs the HSV values on an input image by an amount no greater than the given input values
def color_jitter(img, h_dash, s_dash, v_dash):
    if h_dash < 0 or h_dash > 360:
        print("Value of hue should be between 0 and 360")
    elif s_dash < 0 or s_dash > 1:
        print("Value of saturation should be betweeen 0 and 1")
    elif v_dash < 0 or v_dash > 1:
        print("Value of value inputs should be betweeen 0 and 1")
    else:
        new_img = np.zeros(img.shape)
        
        h_dash = random.uniform(0,h_dash)
        s_dash = random.uniform(0,s_dash)
        v_dash = random.uniform(0,v_dash)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                h,s,v = convert_to_hsv(img[i][j])
                r,g,b = convert_to_rgb(h + h_dash,s + s_dash,v + v_dash)
                new_img[i][j][0] = r
                new_img[i][j][1] = g
                new_img[i][j][2] = b
        
#        hsv_img = rgb_to_hsv(img,h+h_dash,s+s_dash,v+v_dash)
#        rgb_img = hsv_to_rgb(hsv_img)
        io.imsave('img_color_jitter.jpeg', new_img)
        print("New img_color_jitter.jpeg added to folder.")

# Uncomment functions in main to transform images and change the parameters to transform to specific values
if __name__ == '__main__':
    img = io.imread("img.jpeg")
#    random_crop(img, 183)
#    extract_patch(img,2)
    resize_img(img,0.025)
#    color_jitter(img, 100, 0.5, 0.5)
    
