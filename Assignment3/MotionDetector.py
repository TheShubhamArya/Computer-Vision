"""
Shubham Arya 1001650536
CSE 4310 Computer Vision Spring 2022 UTA
Assignment 3 - Kalman filters
Run command: python3 ObjectTracking.py
"""

import numpy as np
import cv2
from skimage.morphology import dilation
from skimage.color import rgb2gray
from skimage.measure import label, regionprops

class MotionDetector():
    # a - Frame hysteresis for determining active or inactive objects.
    # motion_thresh - The motion threshold for filtering out noise.
    # dist_thresh - A distance threshold to determine if an object candidate belongs to an object currently being tracked.
    # s - The number of frames to skip between detection.
    # N - The number of maximum objects to detect.

    def __init__(self, a, motion_thresh, dist_thresh, s, N,KF):
        self.a = a
        self.motion_thresh = motion_thresh
        self.dist_thresh = dist_thresh
        self.s = s
        self.N = N
        self.KF = KF
        
    def detect(self, frames, i):
        s = self.s
        if (i + s > len(frames)) or (i + 2*s > len(frames)):
            return None
        # convert frames from rgb to gray
        ppframe = rgb2gray(frames[i])
        pframe = rgb2gray(frames[i+s])
        cframe = rgb2gray(frames[i+2*s])
        
        # take difference between the frames aand find minimum
        diff1 = np.abs(cframe - pframe)
        diff2 = np.abs(pframe - ppframe)
        motion_frame = np.minimum(diff1, diff2)
        
        thresh_frame = motion_frame > 0.05
        dilated_frame = dilation(thresh_frame, np.ones((9, 9)))
        label_frame = label(dilated_frame)
        regions = regionprops(label_frame)
        
        centers=[]
        for region in regions:
            # if maximum number of objects detected, then stop
            if len(centers) >= self.N:
                break
            minr, minc, maxr, maxc = region.bbox
            area = (maxr-minr)*(maxc-minc)
            
            # Area of the object is betweem 100-1000 pixels
            if area >= 100 and area <= 1000:
                (x1,y1) = self.KF.predict()
                (x2,y2) = self.KF.update([[(minr+maxr)/2], [(minc+maxc)/2]])

                # If the distance between an object proposal and the prediction of one of the filters is less than distance threshold
                if abs(x2-x1) <= self.dist_thresh and abs(y2-y1) <= self.dist_thresh:
#                    print(x1,y1,x2,y2)
                    color = (0, 0, 255) # blue color
                    thickness = 2
                    startPoint = (minc, minr)
                    endPoint = (maxc, maxr)
                    cv2.rectangle(frames[i], startPoint, endPoint, color,thickness)
                    centers.append(np.array(region.bbox))

