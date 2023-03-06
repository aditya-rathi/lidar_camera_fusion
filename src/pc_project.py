#!usr/bin/env python3

import rospy
import tf2_ros
import pcl_ros
from sensor_msgs.msg import Image, PointCloud2
import ros_numpy
import numpy as np
import cv2
from cv_bridge import CvBridge
import threading

class pc_project:
    def __init__(self):
        self.sub1 = rospy.Subscriber("/theia/front_camera/color/image_raw",Image,self.cam_callback)
        self.sub2 = rospy.Subscriber("/theia/os_cloud_node/points",PointCloud2,self.lidar_callback)
        # self.sub2 = rospy.Subscriber("/pc_interpoled",PointCloud2,self.lidar_callback)
        self.pub1 = rospy.Publisher('/combined_output',Image,queue_size=1)
        self.image = None
        self.pcl_arr = None
        self.cam_intrinsic = np.array([[615.3355712890625, 0.0, 333.37738037109375], [0.0, 615.457763671875, 233.50408935546875], [0.0, 0.0, 1.0]])
        self.RT0 = np.array([[0,-1,0,0], [0,0,-1, -0.435],[1,0,0, -0.324],[0, 0, 0, 1]],dtype=np.float32)
        # self.RT0 = np.array([[0,-1,0,0], [0,0,-1, 0],[1,0,0, 0],[0, 0, 0, 1]],dtype=np.float32)
    #     self.RT1 = np.array([[ 0.9878004 ,  0.06272954,  0.14253163, -0.01150406],
    #    [-0.0715505 ,  0.995771  ,  0.05762491, -0.01063806],
    #    [-0.1383141 , -0.06712012,  0.9881114 , -0.00219538],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]],dtype=np.float32)
    #     self.RT2 = np.array([[ 0.9889161 , -0.09491793,  0.1141732 , -0.00490772],
    #    [ 0.0785354 ,  0.98698986,  0.14029662,  0.06105515],
    #    [-0.12600446, -0.12977494,  0.9835046 ,  0.07642391],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]],dtype=np.float32)
    #     self.RT3 = np.array([[ 9.9757671e-01, -5.4184329e-02, -4.3643501e-02,  1.0032160e-04],
    #    [ 5.3290997e-02,  9.9835014e-01, -2.1379454e-02,  1.2742449e-02],
    #    [ 4.4729929e-02,  1.9001840e-02,  9.9881840e-01, -5.5277901e-04],
    #    [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],dtype=np.float32)
    #     self.RT4 = np.array([[ 0.99996215, -0.00632162,  0.00597615, -0.00200605],
    #    [ 0.00648532,  0.999593  , -0.02778118,  0.00685231],
    #    [-0.0057981 ,  0.02781889,  0.9995962 , -0.00794246],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]],dtype=np.float32)
    #     self.RT5 = np.array([[ 9.9995494e-01,  4.8891795e-03,  8.1361691e-03, -2.4311836e-03],
    #    [-4.9085659e-03,  9.9998516e-01,  2.3645191e-03, -2.8120556e-03],
    #    [-8.1244884e-03, -2.4043494e-03,  9.9996406e-01,  4.8492674e-04],
    #    [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],dtype=np.float32)
        self.RTs = []
        for i in range(5):
            self.RTs.append(np.genfromtxt('./src/lidar_camera_fusion/cfg/RT_%d.csv'%i,delimiter=','))
        self.bridge = CvBridge()
        self.rate = rospy.Rate(30)
        self.lock = threading.Lock()
        self.lock2 = threading.Lock()

    def cam_callback(self,data):
        data = ros_numpy.numpify(data)
        self.lock2.acquire()
        # self.image = cv2.cvtColor(data,cv2.COLOR_BGR2RGB)
        self.image = data
        self.lock2.release()
        
 
    def lidar_callback(self,data):
        data = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data,remove_nans=True)
        data = np.hstack((data,np.ones((data.shape[0],1))))
        data = data.T
        data = self.RT0 @ data
        for RT in self.RTs:
            data = np.linalg.inv(RT) @ data
        # data = np.linalg.inv(self.RT1) @ data
        # data = np.linalg.inv(self.RT2) @ data
        # data = np.linalg.inv(self.RT3) @ data
        # data = np.linalg.inv(self.RT4) @ data
        # data = np.linalg.inv(self.RT5) @ data
        data = data.T
        self.lock.acquire()
        self.pcl_arr = data[:,:3]
        self.lock.release()

    def project_depth(self):
        while self.image is None or self.pcl_arr is None:
            rospy.sleep(0.1)
        img_shape = self.image.shape
        self.lock.acquire()
        pcl_xyz = self.cam_intrinsic @ self.pcl_arr.T
        self.lock.release()
        pcl_xyz = pcl_xyz.T
        pcl_z = pcl_xyz[:, 2]
        pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
        pcl_uv = pcl_xyz[:, :2]
        mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (
            pcl_uv[:, 1] < img_shape[0]) & (pcl_z > 0)
        pcl_uv = pcl_uv[mask]
        pcl_z = pcl_z[mask]
        pcl_uv = pcl_uv.astype(np.uint32)
        pcl_z = pcl_z.reshape(-1, 1)
        depth_img = np.zeros((img_shape[0], img_shape[1], 1))
        depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z
        depth_img = ((depth_img/np.max(depth_img))*255).astype(np.uint8)
        depth_img = cv2.applyColorMap(depth_img,cv2.COLORMAP_JET)
        depth_img[np.all(depth_img == [128, 0, 0], axis=-1)] = (0,0,0)
        depth_img = cv2.cvtColor(depth_img,cv2.COLOR_BGR2RGB)     
        s = np.sum(depth_img, axis=2)
        non_black = (s != 0) 
        x, y = np.where(non_black)
        self.lock2.acquire()
        output_img = self.image
        self.lock2.release()
        output_img[x,y] = depth_img[x,y]
        rospy.loginfo("Output published")
        self.pub1.publish(self.bridge.cv2_to_imgmsg(output_img,encoding="rgb8"))
        self.rate.sleep()
        

if __name__ == "__main__":
    rospy.init_node("depth_projector")
    my_obj = pc_project()
    while not rospy.is_shutdown():
        my_obj.project_depth()
    print("shutdown called")

