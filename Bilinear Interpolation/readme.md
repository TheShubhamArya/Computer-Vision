# Bilinear Interpolation

Bilinear interpolation is a sampling strategy which is crucial in arranging pixels while resizing an image. This type of sampling takes into account all the neighboring pixels in the original image. Instead of just taking the pixel values from the pixel closest to in original image, like Nearest Neighboring Interpolation, this takes values from the surrounding pixels by comparing their distance and evaluating their intensities. 

This helps in smoothening the resized image without appearing pixelated. The nearest neighbor on the other hand woul resize image by a factor. Thhis results in a highly pixelated resized image if the orignal image was pixelated or the resize factor was substantial. 

## How it works
Essentially, we put the original image on top of the resized image and then get the neighboring pixel values in the orginal image for our resized image. Once we know what the neighboring pixels in original image for the resized image, we can calculate the intensity of colors we need from the surrounding pixels by calculating their distance. 

So, if a pixel is closer to the top left pixel, then the top left pixel will have a higher intensity and a higher contribution in the resized image. All the other pixels will have some contributions as well depending on their distance. 

## Example
Here is a 100px x 100px image of spongbob squarepant. As you can see, it is already very pixelated.

<!-- ![img_test_x0 1](https://user-images.githubusercontent.com/60827845/153077168-5a71a3b3-7c66-404c-b686-2ea00e0479e7.png) -->
<figure class="image">
  <figcaption><b>Original Image</b> (100px x 100px) resized to 500x500 for display</figcaption>
  <img src="https://user-images.githubusercontent.com/60827845/153077168-5a71a3b3-7c66-404c-b686-2ea00e0479e7.png" width="500" height="500" />
</figure>

Now, if we want to resize the image by a factor of 10, the size of the new image will be 1000px x 1000px. To see how well bilinear interpolation works, let's first see the output nearest neighbor gives us. 

<!-- ![img_test_x10](https://user-images.githubusercontent.com/60827845/153077239-9ee39a57-aecf-4df0-9d44-8c148e7c54af.png) -->
<figure class="image">
  <figcaption><b>Nearest neighbor</b> (1000px x 1000px) resized to 500x500 for display</figcaption>
  <img src="https://user-images.githubusercontent.com/60827845/153077239-9ee39a57-aecf-4df0-9d44-8c148e7c54af.png" width="500" height="500" />
</figure>

As you can see, nearest neighbor gives an even more pixelated output. Basically, it has just mapped RGB values to the reesized image by simple scaling.

Here is what Bilinear Interpolation gives us. You can see that this is a much smoother image compared to the previous one.

<!-- ![bilinear_img_test_x10](https://user-images.githubusercontent.com/60827845/153077256-d4523765-0e37-42e2-83c0-ada1eaa8ed0c.png) -->
<figure class="image">
  <figcaption><b>Bilinear Interpolation</b> (1000px x 1000px) resized to 500x500 for display</figcaption>
  <img src="https://user-images.githubusercontent.com/60827845/153077256-d4523765-0e37-42e2-83c0-ada1eaa8ed0c.png" width="500" height="500" />
</figure>

