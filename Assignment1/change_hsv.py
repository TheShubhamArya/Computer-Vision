"""
Shubham Arya 1001650536
CSE 4310 Computer Vision Spring 2022 UTA
Assignment 1 - Color spaces
"""
import numpy as np
import math
import sys
import skimage.io as io
import skimage.color as color
from skimage import transform

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
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = float(r*255), float(g*255), float(b*255)
    return r, g, b

# For testing only
"""
def rgb_to_hsv(img,h,s,v):
    img_hsv = color.rgb2hsv(img)
    img_hsv[:, :, 0] += (h/360)
    img_hsv[:, :, 1] += s
    img_hsv[:, :, 2] += v
    return img_hsv

def hsv_to_rgb(img):
    img_rgb = color.hsv2rgb(img)
    return img_rgb
"""

def color_spaces(filename,h_dash,s_dash,v_dash):
    if h_dash < 0 or h_dash > 360:
        print("Value of hue should be between 0 and 360")
    elif s_dash < 0 or s_dash > 1:
        print("Value of saturation should be betweeen 0 and 1")
    elif v_dash < 0 or v_dash > 1:
        print("Value of value inputs should be betweeen 0 and 1")
    else:
        img = io.imread(filename)
        new_img = np.zeros(img.shape)
        # loops through each pixel to change it's rgb value by converting it to hsv, altering the hsv, and then converting it back to rgb
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                h,s,v = convert_to_hsv(img[i][j])
                r,g,b = convert_to_rgb(h+h_dash,s+s_dash,v+v_dash)
                new_img[i][j][0] = r
                new_img[i][j][1] = g
                new_img[i][j][2] = b
        
#        hsv_img = rgb_to_hsv(img,h_dash,s_dash,v_dash)
#        rgb_img = hsv_to_rgb(hsv_img)
        io.imsave('new_img.jpeg',new_img)
        print("new_image.jpeg saved to current folder")

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Error: Incorrect format!\nThe run command is: python3 change_hsv.py [filename] [hue value] [saturation value] [value modification]")
    else:
        args = sys.argv
        color_spaces(args[1], float(args[2]), float(args[3]), float(args[4]))
