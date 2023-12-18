#!/usr/bin/env python3

import cv2
import numpy as np
from threading import Thread, Lock
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from markertracker import MarkerTracker

CAMERA_CAPTURE_FREQUENCY = 25 # tested, do not increase!


# OPTICAL FLOW CONFIG
COMPUTE_OF = True # compute optical flow
WIDTH = 240
HEIGHT = 320
RESIZED_FOR_OF = True # set true for optical flow


# # HIGH QUALITY IMAGE CONFIG
# COMPUTE_OF = False
# WIDTH = 2464
# HEIGHT = 3280
# RESIZED_FOR_OF = False # set true for optical flow

cvbridge = CvBridge()

def search_for_devices(total_ids = 10):
    '''
    Found gelsights for the sensor.
    '''

    found_ids = []

    print("Looking for gelsight devices")
    for _id in range(0, total_ids, 2):
        try:
            cap = cv2.VideoCapture(_id)
            ret, _ = cap.read()
            if ret:
                found_ids.append(_id)
                cap.release()
            else:
                break
        except:
            pass

    print('Found {} gelsight sensors.'.format(len(found_ids)))

    return found_ids

def resize_crop_mini(img, imgw, imgh):
    # remove 1/7th of border from each size
    border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(np.floor(img.shape[1] * (1 / 7)))
    # keep the ratio the same as the original image size
    img = img[border_size_x+2:img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
    # final resize for 3d
    img = cv2.resize(img, (imgw, imgh))
    return img

class WebcamVideoStream :
    '''
    Thread based visualization
    '''
    def __init__(self, src, width = 320, height = 240) :
        self.stream = cv2.VideoCapture(src)
        # self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
        # self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        assert self.grabbed != False, "Camera with src={} is not found".format(src)

        self.started = False
        self.read_lock = Lock()

        self.connection_lost = False
        self.src = src

    def start(self) :
        if self.started :
            print ("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame

            if self.frame is None:
                self.connection_lost = True
                print("lost connection with camera, src={}".format(self.src))
                print("Shutting down")
                exit()
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()

class GS:
    def __init__(self, src, width=240, height=320, resized_for_OF=True):
        self.src = src
        self.pub = rospy.Publisher("/gsmini_rawimg_{}".format(src), Image, queue_size=1)
        self.vs = WebcamVideoStream(src=src).start()
        self.width = width
        self.height = height

        self.img = None
        self.OF = None
        self.resized_for_OF = resized_for_OF

        # flash out black pixels at the beginning
        self.flash_out_size=20
        self.initialize()


    def initialize(self):
        for i in range(self.flash_out_size):
            self.vs.read()

    def capture(self):
        img = self.vs.read()
        if self.resized_for_OF:
            self.img = resize_crop_mini(img,self.height,self.width)
        else:
            self.img = cv2.resize(img, (self.width, self.height))
        


    def publish(self):
        img_msg = cvbridge.cv2_to_imgmsg(self.img, encoding="passthrough")
        img_msg.header.stamp = rospy.Time.now()
        img_msg.header.frame_id = 'map'
        self.pub.publish(img_msg)

    
class OpticalFlowDetector:
    def __init__(self, id=0):
        self.id = id
        self.initialized = False

        self.pre_grey_img = None
        self.curr_grey_img = None
        self.p0 = None
        self.nct = None
        self.Ox = None
        self.Oy = None
        # self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def init(self, img):
        '''
        Initialize the markers in the beginning, quite slow process. Takes up to 5 seconds to compute
        '''
        if not self.initialized:

            if self.pre_grey_img is None:
                self.pre_grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img = np.float32(img) / 255.0

            mtracker = MarkerTracker(img)
            marker_centers = mtracker.initial_marker_center
            Ox = marker_centers[:, 1]
            Oy = marker_centers[:, 0]
            nct = len(marker_centers)
            print("Initial markers are detected")

            

            # Existing p0 array
            p0 = np.array([[Ox[0], Oy[0]]], np.float32).reshape(-1, 1, 2)
            for i in range(nct - 1):
                # New point to be added
                new_point = np.array([[Ox[i+1], Oy[i+1]]], np.float32).reshape(-1, 1, 2)
                # Append new point to p0
                p0 = np.append(p0, new_point, axis=0)

            self.p0 = p0
            self.nct = nct
            self.Ox = Ox
            self.Oy = Oy
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

        # Draw the tracks
        flow_lines = {
            'start': [],
            'end': [],
            'size': len(good_new)
        }
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            ix = int(self.Ox[i])
            iy = int(self.Oy[i])

            flow_lines['start'].append((ix, iy))
            flow_lines['end'].append((int(a),int(b)))

        return flow_lines


def calc_force(flow_lines, mag_eps=1.0):
    
    # build the matrix for initial
    init_xy = np.array(flow_lines['start'])
    curr_xy = np.array(flow_lines['end'])

    accepted_mag = []
    accepted_ang = []
    for i in range(flow_lines['size']):
        mag =  init_xy[i] - curr_xy[i]

        accepted_mag.append(mag)
        accepted_ang.append(np.arctan2(init_xy[i], curr_xy[i]))

    force_mag = np.array(accepted_mag).sum(axis=0)
    force_angle = np.arctan2(force_mag[0], force_mag[1])
    force_mag = np.sqrt( (force_mag**2).sum() )

    return force_mag, force_angle*180/np.pi



if __name__ == '__main__':

    rospy.init_node('GS', anonymous=True)

    gs_ids = search_for_devices()

    r = rospy.Rate(CAMERA_CAPTURE_FREQUENCY)
    NUM_SENSORS = len(gs_ids)
    color = np.random.randint(0, 255, (100, 3))

    if NUM_SENSORS == 0:
        print('No gelsight sensor is found! Exiting ...')
        exit()
    # run sensors
    gss = []
    for src in gs_ids:
        gss.append(GS(src, width=WIDTH, height=HEIGHT))

    
    # intialize the optical flow
    if COMPUTE_OF:
        print('Initializing Optical Flow ... ')
        for gs in gss:
            of = OpticalFlowDetector(id=gs.src)
            gs.capture()
            of.init(gs.img)
            gs.OF = of
        print('Done.')

    # run infinity loop
    while not rospy.is_shutdown():
        for gs in gss:
            gs.capture()
            gs.publish()

            if COMPUTE_OF:

                # calculate optical flow - MUST be fast!
                flow_lines = gs.OF.update(gs.img)

                # calc force
                force_mag, force_angle = calc_force(flow_lines)
                print(force_mag)
                print(force_angle)
                
                for i in range(flow_lines['size']):
                    offrame = cv2.arrowedLine(gs.img, flow_lines['start'][i], flow_lines['end'][i], (255,255,255), thickness=1, line_type=cv2.LINE_8, tipLength=.15)
                    # offrame = cv2.circle(offrame, flow_lines['end'][i], 5, color[i].tolist(), -1)

                cv2.imshow('gsmini{}_OF'.format(gs.src), offrame)
                #cv2.imshow('gsmini{}_OF'.format(gs.src), gs.OF.curr_grey_img)

            else:
                cv2.imshow('gsmini{}'.format(gs.src), gs.img)

        if cv2.waitKey(1) == 27 :
            break

        r.sleep()

    # close all windows
    for gs in gss:
        gs.vs.stop()

    cv2.destroyAllWindows()
