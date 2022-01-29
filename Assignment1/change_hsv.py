import numpy as np
import sys
import skimage.io as io
import skimage.color as color
from skimage import transform

def rgb_to_hsv(img,h,s,v):
    img_hsv = color.rgb2hsv(img)
    img_hsv[:, :, 0] += h
    img_hsv[:, :, 1] += s
    img_hsv[:, :, 2] += v
    return img_hsv

def hsv_to_rgb(img):
    img_rgb = color.hsv2rgb(img)
    return img_rgb

def color_spaces(filename,h,s,v):
    if h < 0 or h > 360:
        print("Value of hue should be between 0 and 360")
    elif s < 0 or s > 1:
        print("Value of saturation should be betweeen 0 and 1")
    elif v < 0 or v > 1:
        print("Value of value inputs should be betweeen 0 and 1")
    else:
        img = io.imread(filename)
        hsv_img = rgb_to_hsv(img,h,s,v)
        rgb_img = hsv_to_rgb(hsv_img)
        io.imsave('new_image.jpeg', rgb_img)

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Error: Incorrect format!\nThe run command is: python3 change_hsv.py [filename] [hue value] [saturation value] [value modification]")
    else:
        args = sys.argv
        color_spaces(args[1], float(args[2]), float(args[3]), float(args[4]))
