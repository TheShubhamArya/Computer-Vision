"""
Shubham Arya
Bilinear interpolation
Run command: python3 img_bilinear_resize.py [filename] [factor]
"""
import numpy as np
import sys
import math
import skimage.io as io

def bilinear_pixel(img, posX, posY):

    x0 = int(posX)
    y0 = int(posY)
    x1 = posX - x0
    y1 = posY - y0
    x0Plus1 = min(x0+1,img.shape[1]-1)
    y0Plus1 = min(y0+1,img.shape[0]-1)
    
    # Gets rgb value at 4 corners in original image
    top_left = img[x0][y0Plus1]
    top_right = img[x0Plus1][y0Plus1]
    bottom_left = img[x0][y0]
    bottom_right = img[x0Plus1][y0]
 
    #Calculate interpolation
    top = x1 * top_right + (1 - x1) * top_left
    bottom = x1 * bottom_right + (1 - x1) * bottom_left
    new_rgb = y1 * top + (1 - y1) * bottom
 
    return new_rgb
    
if __name__== '__main__':

    if len(sys.argv) != 3:
        print("Error: Incorrect format!\nThe run command is: python3 img_bilinear_resize.py [filename] [factor]")
    else:
        args = sys.argv
        filename = args[1]
        factor = float(args[2])
        img = io.imread(filename)
        
        row = img.shape[0]
        col = img.shape[1]
        new_row = row*factor
        new_col = col*factor
        
        new_img = np.zeros([int(new_row),int(new_col),img.shape[2]])
        rowScale = float(row)/float(new_row)
        colScale = float(col)/float(new_col)
     
        for i in range(int(new_row)):
            for j in range(int(new_col)):
                posX = i * rowScale
                posY = j * colScale
                new_img[i][j] = bilinear_pixel(img, posX, posY)
     
        filename = filename+"_x"+str(factor)+".png"
        io.imsave(filename, new_img)
        print("Resized image saved to folder as ",filename)
    
