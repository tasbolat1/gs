#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
from markertracker import MarkerTracker
import time


# TODO:
# 1. receive the images - Done
# 2. Calculate the optical flow - Done
# 3. calculate the force


class OpticalFlowDetector:
    def __init__(self, id=0):
        self.id = id
        self.initialized = False

        self.pre_grey_img = None
        self.curr_grey_img = None
        self.p0 = None
        self.nct = None
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
    def init(self, img):
        '''
        Initialize the markers in the beginning, quite slow process. Takes up to 5 seconds to compute
        '''
        if not self.initialized:
            mtracker = MarkerTracker(img)
            marker_centers = mtracker.initial_marker_center
            Ox = marker_centers[:, 1]
            Oy = marker_centers[:, 0]
            nct = len(marker_centers)
            print("Initial markers are detected")

            if self.pre_grey_img is None:
                self.pre_grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Existing p0 array
            p0 = np.array([[Ox[0], Oy[0]]], np.float32).reshape(-1, 1, 2)
            for i in range(nct - 1):
                # New point to be added
                new_point = np.array([[Ox[i+1], Oy[i+1]]], np.float32).reshape(-1, 1, 2)
                # Append new point to p0
                p0 = np.append(p0, new_point, axis=0)

            self.p0 = p0
            self.nct = nct
            self.initialized = True



    def update(self, img):

        if not self.initialized:
            self.init(img)
            return

        self.curr_grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(self.pre_grey_img, self.curr_grey_img, self.p0, None, **self.lk_params)
        
        # Select good points
        good_new = p1[st == 1]
        good_old = self.p0[st == 1]

        if len(good_new) < self.nct:
            # Detect new features in the current frame
            print(f"all pts did not converge")
        else:
            # Update points for next iteration
            self.p0 = good_new.reshape(-1, 1, 2)
        
        self.pre_grey_img = self.curr_grey_img.copy()
        

class GS:
    def __init__(self):
        rospy.Subscriber("/gsmini_rawimg_0", Image, self.gsCallback, queue_size=1)

        self.cvbridge = CvBridge()

        self.opt_flow0 = OpticalFlowDetector(0)    

    def gsCallback(self, msg):
        # convert to cv image
        img = self.cvbridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        # run optical flow
        sss = time.time()
        self.opt_flow0.update(img)
        print('Time', time.time()-sss)
    

if __name__ == '__main__':
    rospy.init_node('gs', anonymous=True)
    gs = GS()

    # rospy.sleep(5)

    # cv2.imshow("grey image", gs.old_gray)


    rospy.spin()