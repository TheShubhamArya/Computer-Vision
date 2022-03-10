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
