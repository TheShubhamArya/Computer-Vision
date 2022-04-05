"""
Shubham Arya 1001650536
CSE 4310 Computer Vision Spring 2022 UTA
Assignment 3 - Kalman filters
Run command: python3 ObjectTracking.py
"""

import cv2
from MotionDetector import MotionDetector
from KalmanFilter import KalmanFilter
import skvideo.io

if __name__ == "__main__":

    """
    To get the best results with the parking.mp4 video, I made following assumptions which is reflected in MotionDetector.py file.
    1) The area of the detected image is between 100 to 1000 pixels.
    2) Maximum distance threshold for predicted points and updated points is 100 pixels.
    3) Maximum number of objects that can be detected is 25. This cn be increased while initializing the MotionDetector class. 
    """

    filepath = input("Enter file path of video: ")

    frames = skvideo.io.vread(filepath)#"parking.mp4")

    # KalmanFilter(dt)
    KF = KalmanFilter(0.1)
    #  MotionDetector(a, motion_thresh, dist_thresh, s, N, KalmanFilter)
    motionDetector = MotionDetector(1,1,100,1,25,KF)
    for i in range(0,len(frames)-2):
        motionDetector.detect(frames,i)
        frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        cv2.imshow('image', frame)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        cv2.waitKey(1)

