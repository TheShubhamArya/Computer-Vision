"""
Shubham Arya 1001650536
CSE 4310 Computer Vision Spring 2022 UTA
Assignment 3 - Kalman filters
Run command: python3 ObjectTracking.py
"""

import numpy as np

# Video reference: https://youtu.be/E-6paM_Iwfc
class KalmanFilter(object):

    def __init__(self, dt):

        self.dt = dt
        
        # Intial State
        # u is the control vector
        self.u = np.matrix([[0],[0]])

        self.x = np.matrix([[0], [0], [0], [0]])

        self.D = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
                            
        # Covariance Matrix
        self.E = np.eye(self.D.shape[1])

        # B is the control matrix
        self.B = np.matrix([[(self.dt**2)/2, 0],
                            [0,(self.dt**2)/2],
                            [self.dt,0],
                            [0,self.dt]])

        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        # Noise Covariance
        self.R = np.matrix([[0.1,0],
                           [0, 0.1]])

       

    def predict(self):
        # x_k = D_k.x_k-1 + B_k.u_k |Predicted mean|
        self.x = np.dot(self.D, self.x) + np.dot(self.B, self.u)

        # E_k = D_k.E_k-1.DT_k |Sigma prediction|
        self.E = np.dot(np.dot(self.D, self.E), self.D.T)
        """
            x_k is the prediction of our current state based on the previous best estimate with an added correction term based on known factors. E_k is the updated uncertainty based on the old uncertainty with added Gaussian noise to reflect unknown factors.
        """
        return int(self.x[0]), int(self.x[1])

    def update(self, z):
        S = np.linalg.inv(np.dot(self.H, np.dot(self.E, self.H.T)) + self.R)
        # Calculate the Kalman Gain
        K = np.dot(np.dot(self.E, self.H.T), S)
        self.x += np.dot(K, (z - np.dot(self.H, self.x)))
        self.E -=  K*self.H*self.E
        return int(self.x[0]), int(self.x[1])

