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

flag = True

class pc_project:
    def __init__(self):
        self.sub1 = rospy.Subscriber("/theia/right_camera/color/image_raw",Image,self.cam_callback)
        self.sub2 = rospy.Subscriber("/theia/os_cloud_node/points",PointCloud2,self.lidar_callback)
        self.pub1 = rospy.Publisher('/combined_output',Image,queue_size=1)
        self.image = None
        self.pcl_arr = None
        self.cam_intrinsic = np.array([[615.3355712890625, 0.0, 333.37738037109375], [0.0, 615.457763671875, 233.50408935546875], [0.0, 0.0, 1.0]])
        self.RT0 = np.array([[0, 1, 0 , 0], [-1, 0, 0, 0],[0, 0, 1, 0],[0, -0.1778, -0.381, 1]],dtype=np.float32)
        self.RT1 = np.array([[ 0.9995117 ,  0.02276611,  0.00481033, -0.09313965],
       [-0.02311707,  0.9472656 ,  0.31982422,  0.88671875],
       [ 0.00272942, -0.32006836,  0.9472656 , -0.37231445],
       [ 0.        ,  0.        ,  0.        ,  1.        ]],dtype=np.float32)
        self.RT2 = np.array([[ 0.99902344, -0.04901123,  0.00145721,  1.1611328 ],
       [ 0.04898071,  0.99609375, -0.07366943,  0.2409668 ],
       [ 0.00216103,  0.07366943,  0.9970703 ,  0.37036133],
       [ 0.        ,  0.        ,  0.        ,  1.        ]],dtype=np.float32)
        self.RT3 = np.array([[ 0.99853516,  0.02578735,  0.04266357,  0.20812988],
       [-0.01991272,  0.99072266, -0.1328125 , -0.15588379],
       [-0.04571533,  0.13183594,  0.9902344 , -0.42163086],
       [ 0.        ,  0.        ,  0.        ,  1.        ]],dtype=np.float32)
        self.RT4 = np.array([[ 1.        ,  0.01309967, -0.00514984, -0.04016113],
       [-0.01330566,  0.99902344, -0.04241943,  0.0378418 ],
       [ 0.00458908,  0.04248047,  0.99902344, -0.07385254],
       [ 0.        ,  0.        ,  0.        ,  1.        ]],dtype=np.float32)
        self.RT5 = np.array([[ 1.0000000e+00,  2.0732880e-03,  3.6168098e-04, -1.4400482e-03],
       [-2.0694733e-03,  1.0000000e+00, -1.1581421e-02,  4.7149658e-02],
       [-3.8576126e-04,  1.1581421e-02,  1.0000000e+00,  1.9550323e-03],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],dtype=np.float32)
        self.bridge = CvBridge()
        self.rate = rospy.Rate(30)
        self.lock = threading.Lock()
        self.lock2 = threading.Lock()

    def cam_callback(self,data):
        data = ros_numpy.numpify(data)
        self.lock2.acquire()
        self.image = cv2.cvtColor(data,cv2.COLOR_BGR2RGB)
        self.lock2.release()
        
 
    def lidar_callback(self,data):
        data = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data,remove_nans=True)
        data = np.hstack((data,np.ones((data.shape[0],1))))
        data = data.T
        data = np.linalg.inv(self.RT0) @ data
        data = np.linalg.inv(self.RT1) @ data
        data = np.linalg.inv(self.RT2) @ data
        data = np.linalg.inv(self.RT3) @ data
        data = np.linalg.inv(self.RT4) @ data
        data = np.linalg.inv(self.RT5) @ data
        data = data.T
        self.lock.acquire()
        self.pcl_arr = data[:,:3]
        self.lock.release()

    def project_depth(self):
        global flag
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
        if flag:
            temp = pcl_uv.copy()
            temp = temp + np.absolute(np.min(temp,axis=0))
            temp = temp.astype(np.uint32)
            print(np.min(temp,axis=0))
            my_img = np.zeros((np.max(temp,axis=0)),dtype=np.uint8)
            my_img[temp[:,1],temp[:,0]] = pcl_z.reshape(-1,1)
            my_img = ((my_img/np.max(my_img))*255).astype(np.uint8)
            cv2.imwrite('temp.png',my_img)
            flag = False
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
        depth_img = cv2.cvtColor(depth_img,cv2.COLOR_BGR2RGB)
        self.lock2.acquire()
        output_img = cv2.addWeighted(self.image,0.2,depth_img,0.8,0)
        self.lock2.release()
        rospy.loginfo("Output published")
        self.pub1.publish(self.bridge.cv2_to_imgmsg(output_img,encoding="rgb8"))
        self.rate.sleep()
        

if __name__ == "__main__":
    rospy.init_node("depth_projector")
    my_obj = pc_project()
    while not rospy.is_shutdown():
        my_obj.project_depth()
    print("shutdown called")

