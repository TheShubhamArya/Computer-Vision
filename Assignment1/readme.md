# Color Spaces
Run command: python3 change_hsv.py [filename] [hue value] [saturation value] [value modification]

To change the Hue, Saturation and Values of an image, we first need to create functions that can convert the RGB values of an image to HSV values and a function to convert the HSV values to RGB values. To make these conversions, I made use of the formulae for RGB to HSV and HSV to RGB.

### RGB to HSV
![RGB to HSV](https://i.stack.imgur.com/FTH7K.png)

### HSV to RGB
![HSV to RGB](https://i.stack.imgur.com/6DMbA.png)

After converting RGB to HSV, we can then adjust the HSV values with the input provided from the command line. Then convert the changed HSV values to RGB and feed it to the newly created image.

# Image Transformations
Run command: python3 img_transforms.py. Image name should be img.jpeg or change the name of file in the code. Uncomment the functions in main function to run through all the different types of image transformation.

## Random crop
Takes the square size of an image and creates a random crop by using the random function. It also makes sure that the coordinates of the new image does not exceed the max coordinates the original image.

## Extract patches
This function creates n^2 patches of an image where n is the input referring to the number of rows an image should be split into. It jumps through blocks of matrices inside the image and forms the patches.

## Resize image
This function resizes the original image by a factor provided by the user. This resizing makes use of nearest neighbor interpolation which assigns a value to pixel depending on the pixel it is closest too when mapped to the pixel in original image.

## Color Jittering
This uses the basics from Color Spaces on the top. The only difference here is that it makes random adjustemeeents with the provided values being the max the image can be adjusted too.

# Image Pyramids
Run command: python3 create_img_pyramid.py. 

Image name should be img.png or change the name of file in the code. The function **resize_img(image_name, height)** saves height - 1 number of images with different sizes which will be saved into the current folder as img_x2.jpeg, img_x4.jpeg, img_x8.jpeg, and so on. This uses the basic functionalities of resizing image from image transformations. 
