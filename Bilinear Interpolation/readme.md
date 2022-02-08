# Bilinear Interpolation

Bilinear interpolation is a sampling strategy which is crucial in arranging pixels while resizing an image. This type of sampling takes into account all the neighboring pixels in the original image. Instead of just taking the pixel values from the pixel closest to in original image, like Nearest Neighboring Interpolation, this takes values from the surrounding pixels by comparing their distance and evaluating their intensities. 

This helps in smoothening the resized image without appearing pixelated. The nearest neighbor on the other hand woul resize image by a factor. Thhis results in a highly pixelated resized image if the orignal image was pixelated or the resize factor was substantial. 

## How it works
Essentially, we put the original image on top of the resized image and then get the neighboring pixel values in the orginal image for our resized image. Once we know what the neighboring pixels in original image for the resized image, we can calculate the intensity of colors we need from the surrounding pixels by calculating their distance. 

So, if a pixel is closer to the top left pixel, then the top left pixel will have a higher intensity and a higher contribution in the resized image. All the other pixels will have some contributions as well depending on their distance. 

## Example
Here is a 100px x 100px image of spongbob squarepant. As you can see, it is already very pixelated.

Now, if we want to resize the image by a factor of 10, the size of the new image will be 1000px x 1000px. To see how well bilinear interpolation works, let's first see the output nearest neighbor gives us. 

As you can see, nearest neighbor gives an even more pixelated output. Basically, it has just mapped RGB values to the reesized image by simple scaling.

Here is what Bilinear Interpolation gives us. You can see that this is a much smoother image compared to the previous one. 
