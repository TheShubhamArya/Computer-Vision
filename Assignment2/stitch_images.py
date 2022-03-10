"""
Shubham Arya 1001650536
CSE 4310 Computer Vision Spring 2022 UTA
Assignment 2 - Image stitching
Run Command- python3 stitch_images.py
"""
import numpy as np
import math
import random
import skimage.io as io
import matplotlib.pyplot as plt
import cv2

# Returns RGB image by reading the path name
def get_image(path):
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

# 1. Keypoint detection using SIFT from OpenCV
def SIFT(img):
    siftDetector = cv2.SIFT_create()
    keypoints, descriptors = siftDetector.detectAndCompute(img, None)
    return keypoints, descriptors
    
# 2. Keypoint matching for 2 images
def keypoint_matching(kp1, des1, kp2, des2):
    # feature matching using brute force
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    good = []
    
    for m,n in matches:
        if m.distance < n.distance:
            good.append([m])
            
    matches = []
    for pair in good:
        matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))
        
    return np.array(matches)

# 2.1 Plot keypoint matches by showing two images with matched keypoints
def plot_keypoint_matches(matches, img1, img2, type):
    print("Displaying ",len(matches)," keypoint matches.")
#    print("img1 ",img1.shape," img2 ", img2.shape)
    if img1.shape[0] != img2.shape[0]:
        if img1.shape[0] > img2.shape[0]:
            img1 = img1[0:img2.shape[0],:,:]
        else:
            img2 = img2[0:img1,shape[0],:,:]
    img = np.concatenate((img1, img2), axis=1)
    if type == "all":
        fig_subtitle = "All Matching Keypoints"
    else :
        fig_subtitle = "Inliers Matching Keypoints"
    
    offset = img.shape[1]/2
    fig, ax = plt.subplots()
    fig.suptitle(fig_subtitle, fontsize=18)
    ax.set_aspect('equal')
    ax.imshow(np.array(img).astype('uint8'))
    
    # Plots 'x' symbol on each keypoint in both the images
    ax.plot(matches[:, 0], matches[:, 1], 'xr')
    ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')
     
    # Creates lines between the two x points
    ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]], 'r', linewidth=0.25)

    plt.show()
    
# Calculates the maybe homograhy model for set of selected points.
def compute_projective_transform(points):
    rows = []
    for i in range(points.shape[0]):
        p1 = np.append(points[i][0:2], 1)
        p2 = np.append(points[i][2:4], 1)
        rows.append([0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]])
        rows.append([p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]])
        
    _, _, V = np.linalg.svd(rows)
    H = np.reshape(V[-1], (3, 3)) # Converts array of size 9 to 2D square matrix of side 3
    H = H/H[2, 2] # To make the element at index 3,3 to 1, and divide all elements with that index
    return H
    
def get_error(points, H):
    num_points = len(points)
    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    all_p2 = points[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p1[i])
        estimate_p2[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
    # Compute error
    errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1)
    return errors
    
"""
3.3 RANSAC - Random Sampling Consensus
Input:
    matches - All the matching points between two images
    n – Minimum number of data points required to estimate model parameters.
    k – Maximum number of iterations allowed in the algorithm.
    t – Threshold value to determine data points that are fit well by model.
    d – Number of close data points required to assert that a model fits well to data.
Output:
    inliers - Returns a subset of matches that have the best fit w.r.t. the model
    H - best fit
"""
def ransac(matches, t, k, n=4, d = 300):
    print("Running RANSAC to remove any outliers.")
    num_best_inliers = 0
    best_inliers = None
    best_H = None
    i = 0
    while i < k:
        idx = random.sample(range(len(matches)), n)
        points = np.array([matches[i] for i in idx ])
        H = compute_projective_transform(points)
            
        errors = get_error(matches, H)
        idx = np.where(errors < t)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > d:
            if num_inliers > num_best_inliers:
                best_inliers = inliers.copy()
                num_best_inliers = num_inliers
                best_H = H.copy()
        i += 1
    print("Now have ",num_best_inliers," matches.")
    return best_inliers, best_H
    
# computes new size of image and warps image based on the input matrix
def warp(x_min, y_min, img, M):
    height_new = int(round(abs(y_min) + img.shape[0]))
    width_new = int(round(abs(x_min) + img.shape[1]))
    size = (width_new, height_new)
    warped = cv2.warpPerspective(src=img, M=M, dsize=size)
    return warped
    
def stitch_images(left, right, H):
    print("Stitching image")
    
    # Normlizes the value of RGB from 0-255 to 0-1
    left = cv2.normalize(left.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    right = cv2.normalize(right.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    
    # transform corners of img1 by the inversee of the best fit model
    rows, cols, _ = left.shape
    corners = np.array([
        [0, 0, 1],
        [cols, 0, 1],
        [0, rows, 1],
        [cols, rows, 1]
    ])
    
    corners_proj = np.dot(H, corners.T)
    
    y_min = min(corners_proj[1] / corners_proj[2])
    x_min = min(corners_proj[0] / corners_proj[2])
    
    translation_mat = np.array([
                                [1, 0, -x_min],
                                [0, 1, -y_min],
                                [0, 0, 1]
                               ])
                               
    warped_r = warp(x_min,y_min,right,translation_mat)
    
    H = np.dot(translation_mat, H)
    warped_l = warp(x_min,y_min,left,H)
    
    # Stitching procedure, store results in warped_l.
    for i in range(warped_r.shape[0]):
        for j in range(warped_r.shape[1]):
            if (warped_l[i][j] == [0,0,0]).all() and (warped_r[i][j] != [0,0,0]).all():
                warped_l[i, j, :] = warped_r[i, j, :]
                  
    stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :]
    return stitch_image
    
    
# Starting point of the image stitching program
if __name__ == '__main__':
    
#    image_paths = [ "a2_images/campus_002.jpg","a2_images/campus_001.jpg","a2_images/campus_000.jpg"]
    image_paths = ["a2_images/yosemite1.jpg", "a2_images/yosemite2.jpg", "a2_images/yosemite3.jpg","a2_images/yosemite4.jpg"]
#    image_paths = ["a2_images/Rainier1.png","a2_images/Rainier2.png"]
    images = []
    for image_path in image_paths:
        # Gets the RGB image values for the image path
        images.append(get_image(image_path))
    
    current_img = images[0]
    counter = 1
    while counter < len(images):
        print("Loop ",counter)
        img1 = current_img
        img2 = images[counter]
        img1_GRAY = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_GRAY = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Keypoint detection using SIFT
        print("Computing keypoint detection in images")
        kp1, des1 = SIFT(img1_GRAY)
        kp2, des2 = SIFT(img2_GRAY)
        
        # Keypoint matching between two images
        print("Matching keypoints between 2 images")
        matches = keypoint_matching(kp1, des1, kp2, des2)
        
        # For display purpose. Shows all the matching keypoints between the 2 images
        plot_keypoint_matches(matches, img1, img2, "all")

        # Ransac to remove the outliers and find the best model with inliers
        inliers, H = ransac(matches, 0.5, len(matches),4,  0)
    
        if H is not None:
            # To display all the keypoints between 2 images that are images
            plot_keypoint_matches(inliers, img1, img2, "inliers")
            
            # Stitching image
            stitched_image = stitch_images(img1, img2, H)
            fig, ax = plt.subplots()
            fig.suptitle("Stitched Image", fontsize=18)
            ax.imshow(stitched_image)
            plt.show()
            
            
            counter += 1
            current_img = (stitched_image * 255).astype('uint8')
            
        else:
            print("Not enough data points for a fit model")
            break
   
    io.imsave("stitched_image.png", stitched_image)
    print("Image stitched_image.png saved in the current folder.")
