# Image Stitching
This assignment required us to stitch multiple images together by:
1) Using SIFT keypoint detection for each image
2) Keypoint matching to match all the similar keypoints between 2 images
3) Finding inliers between these matched keypoints using RANSAC (RANdon SAmpling Consensus)
4) Warping images based on these inliers by generating affine and projective transformations
5) Stitching the warped images together to create one stitched image.

## Requirements
This assignment uses OpenCV, Matplotlib, and Numpy libraries to perform the core functionalities. 
For this assignment, I did not create my own SIFT keypoint detection algorithm or warping algorithm. Instead, I used the functions provided by OpenCV to do this.

Another important requirement for successful stitching of images is that the images should be appended in order, that is, the image that appears on the left should be appended first followed by the rest. 

## Run Command
python3 stitch_images.py

In the stitch_images.py, enter the file path of the images as a string.

## 1. SIFT
SIFT stands for scale invariant feature transform. This finds important keypoints in an image which do not change with the change in size or orientation. To find these keypoints, we use the OpenCV feature detector to get keypoints for our image. After this is done, we get an image like the one below.

 <table>
  <tr>
    <td><img width="578" alt="Screen Shot 2022-03-10 at 6 47 05 PM" src="https://user-images.githubusercontent.com/60827845/157780117-74039e5c-75d3-4bd4-989a-32ec74e665c3.png"></td>
    <td><img width="570" alt="Screen Shot 2022-03-10 at 6 53 31 PM" src="https://user-images.githubusercontent.com/60827845/157780693-63817cdd-30d2-48b0-b9ad-f2ac1e818b1a.png"></td>
  </tr>
 </table>


## 2. Keypoint matching
From all the keypoints that we got in the two images, we match the keypoints that are the same. 
<img width="572" alt="Screen Shot 2022-03-10 at 6 54 39 PM" src="https://user-images.githubusercontent.com/60827845/157780805-9fbd88d2-8479-4bf3-8fc6-f884c1acdd3f.png">

## 3. RANSAC
RANSAC randomly selects a number of keypoints between the two images, then finds the line that fits these points and checks how many points fall on or near the line. This is repeated for a number of times which can be changed,
<img width="569" alt="Screen Shot 2022-03-10 at 6 55 46 PM" src="https://user-images.githubusercontent.com/60827845/157780928-f79b090f-3c6a-4c4f-b802-3b628d933b0f.png">

## 4. Warping Images

## 5. Image Stitching
<img width="559" alt="Screen Shot 2022-03-10 at 6 56 40 PM" src="https://user-images.githubusercontent.com/60827845/157781027-16fd430f-d03e-4889-b87e-b2816ef204dd.png">



