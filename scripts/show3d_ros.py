import sys
import numpy as np
import cv2
import os
import open3d
import copy
import gs3drecon
import rospy
from sensor_msgs.msg import PointCloud2, Image
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2
from cv_bridge import CvBridge, CvBridgeError
from threading import Thread, Lock


cvbridge = CvBridge()

WIDTH = 240
HEIGHT = 320


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
            try:
                (grabbed, frame) = self.stream.read()
                self.read_lock.acquire()
                self.grabbed, self.frame = grabbed, frame

                if self.frame is None:
                    self.connection_lost = True
                    print("lost connection with camera, src={}".format(self.src))
                    print("Shutting down")
                    rospy.signal_shutdown("Connection Lost")
                self.read_lock.release()

            except:
                self.connection_lost = True
                print("Smth happened with camera, src={}".format(self.src))
                print("Shutting down")
                rospy.signal_shutdown("Smth bad happened")

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
        self.pre_img = None
        self.saved_img=None

        # Optical filter holder
        self.OF = None
        self.resized_for_OF = resized_for_OF
        
        # Force estimate holder
        self.FE = None

        # flash out black pixels at the beginning
        self.flash_out_size=50
        self.initialize()

        
    def save_image_instance(self):
        print('saving ...')
        self.saved_img = self.img.copy()

    def initialize(self):
        for i in range(self.flash_out_size):
            self.vs.read()

    def capture(self):

        img = self.vs.read()

        # if self.pre_img is None:
        #     if self.resized_for_OF:
        #         self.pre_img = resize_crop_mini(img,self.height,self.width)
        #     else:
        #         self.pre_img = cv2.resize(img, (self.width, self.height))
        # else:
        #     self.pre_img = self.img.copy()

        if self.resized_for_OF:
            self.img = resize_crop_mini(img,self.height,self.width)
        else:
            self.img = cv2.resize(img, (self.width, self.height))

        

    def publish(self):
        img_msg = cvbridge.cv2_to_imgmsg(self.img, encoding="passthrough")
        img_msg.header.stamp = rospy.Time.now()
        img_msg.header.frame_id = 'map'
        self.pub.publish(img_msg)


def get_diff_img(img1, img2):
    return np.clip((img1.astype(int) - img2.astype(int)), 0, 255).astype(np.uint8)


def get_diff_img_2(img1, img2):
    return (img1 * 1.0 - img2) / 255. + 0.5


def main(argv):

    rospy.init_node('showmini3dros', anonymous=True)

    # Set flags
    SAVE_VIDEO_FLAG = False
    GPU = False
    MASK_MARKERS_FLAG = True
    USE_ROI = False
    PUBLISH_ROS_PC = True
    SHOW_3D_NOW = True
    # Path to 3d model
    path = '.'

    # Set the camera resolution
    mmpp = 0.0634  # mini gel 18x24mm at 240x320

    # This is meters per pixel that is used for ros visualization
    mpp = mmpp / 1000.

    # the device ID can change after chaning the usb ports.
    # on linux run, v4l2-ctl --list-devices, in the terminal to get the device ID for camera
    # dev = gsdevice.Camera("GelSight Mini")
    gs = GS(src=0, width=WIDTH, height=HEIGHT)
    net_file_path = 'nnmini.pt'

    ''' Load neural network '''
    model_file_path = path
    net_path = os.path.join(model_file_path, net_file_path)
    print('net path = ', net_path)

    gpuorcpu = "cpu"

    nn = gs3drecon.Reconstruction3D(imgw=WIDTH, imgh=HEIGHT)
    net = nn.load_nn(net_path, gpuorcpu)

    gs.capture()
    f0 = gs.img

    if SAVE_VIDEO_FLAG:
        #### Below VideoWriter object will create a frame of above defined The output is stored in 'filename.avi' file.
        file_path = './3dnnlive.mov'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(file_path, fourcc, 60, (f0.shape[1],f0.shape[0]), isColor=True)

    if PUBLISH_ROS_PC:
        ''' ros point cloud initialization '''
        x = np.arange(WIDTH) * mpp
        y = np.arange(HEIGHT) * mpp
        X, Y = np.meshgrid(x, y)
        points = np.zeros([WIDTH * HEIGHT, 3])
        points[:, 0] = np.ndarray.flatten(X)
        points[:, 1] = np.ndarray.flatten(Y)
        Z = np.zeros((HEIGHT, WIDTH))  # initialize points array with zero depth values
        points[:, 2] = np.ndarray.flatten(Z)
        gelpcd = open3d.geometry.PointCloud()
        gelpcd.points = open3d.utility.Vector3dVector(points)
        gelpcd_pub = rospy.Publisher("/gsmini_pcd", PointCloud2, queue_size=10)

    if USE_ROI:
        roi = cv2.selectROI(f0)
        roi_cropped = f0[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        cv2.imshow('ROI', roi_cropped)
        print('Press q in ROI image to continue')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('roi = ', roi)

    print('press q on image to exit')

    ''' use this to plot just the 3d '''
    if SHOW_3D_NOW:
        vis3d = gs3drecon.Visualize3D(HEIGHT, WIDTH, '', mmpp)

    try:
        rate = rospy.Rate(60)
        while not rospy.is_shutdown():

            # get the roi image
            gs.capture()
            f1 = gs.img
            if USE_ROI:
                f1 = f1[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
            bigframe = cv2.resize(f1, (f1.shape[1] * 2, f1.shape[0] * 2))
            #cv2.imshow('Image', bigframe)

            # compute the depth map
            dm = nn.get_depthmap(f1, MASK_MARKERS_FLAG)

            ''' Display the results '''
            if SHOW_3D_NOW:
                vis3d.update(dm)

            if PUBLISH_ROS_PC:
                print ('publishing ros point cloud')
                dm_ros = copy.deepcopy(dm) * mpp
                ''' publish point clouds '''
                header = std_msgs.msg.Header()
                header.stamp = rospy.Time.now()
                header.frame_id = 'gs_mini'
                points[:, 2] = np.ndarray.flatten(dm_ros)
                gelpcd.points = open3d.utility.Vector3dVector(points)
                gelpcdros = pcl2.create_cloud_xyz32(header, np.asarray(gelpcd.points))
                gelpcd_pub.publish(gelpcdros)

            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
            if SAVE_VIDEO_FLAG:
                out.write(f1)

            rate.sleep()

    except KeyboardInterrupt:
        print('Interrupted!')


if __name__ == "__main__":
    main(sys.argv[1:])
