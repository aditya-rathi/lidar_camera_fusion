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
        # self.sub2 = rospy.Subscriber("/theia/os_cloud_node/points",PointCloud2,self.lidar_callback)
        self.sub2 = rospy.Subscriber("/pc_interpoled",PointCloud2,self.lidar_callback)
        self.pub1 = rospy.Publisher('/combined_output',Image,queue_size=1)
        self.image = None
        self.pcl_arr = None
        self.cam_intrinsic = np.array([[615.3355712890625, 0.0, 333.37738037109375], [0.0, 615.457763671875, 233.50408935546875], [0.0, 0.0, 1.0]])
        self.RT0 = np.array([[0,-1,0,0], [0,0,-1, -0.435],[1,0,0, -0.324],[0, 0, 0, 1]],dtype=np.float32)
        self.RT1 = np.array([[ 0.99658203,  0.06451416, -0.05041504,  4.765625  ],
       [-0.07531738,  0.9633789 , -0.2565918 , -0.24169922],
       [ 0.03201294,  0.25952148,  0.96533203,  1.4726562 ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]],dtype=np.float32)
        self.RT2 = np.array([[ 1.0000000e+00, -5.8326721e-03, -1.4747620e-02,  3.4003906e+00],
       [ 3.0860901e-03,  9.8388672e-01, -1.7980957e-01, -5.6915283e-02],
       [ 1.5556335e-02,  1.7980957e-01,  9.8339844e-01,  1.4101562e+00],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],dtype=np.float32)
        self.RT3 = np.array([[ 0.99658203, -0.07879639,  0.01820374,  0.6269531 ],
       [ 0.08032227,  0.99121094, -0.10559082, -0.62841797],
       [-0.00971222,  0.10675049,  0.9941406 , -0.32763672],
       [ 0.        ,  0.        ,  0.        ,  1.        ]],dtype=np.float32)
        self.RT4 = np.array([[ 0.99902344, -0.04492188, -0.01132202,  0.01939392],
       [ 0.04452515,  0.99853516, -0.03213501,  0.1574707 ],
       [ 0.01274872,  0.03161621,  0.9995117 , -0.19995117],
       [ 0.        ,  0.        ,  0.        ,  1.        ]],dtype=np.float32)
        self.RT5 = np.array([[ 9.99511719e-01, -2.25219727e-02,  8.42285156e-03,-7.63702393e-03],
       [ 2.26440430e-02,  9.99511719e-01, -1.35650635e-02, 2.02026367e-01],
       [-8.11767578e-03,  1.37557983e-02,  1.00000000e+00, 4.58061695e-05],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]],dtype=np.float32)
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
        data = self.RT0 @ data
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
        depth_img = cv2.cvtColor(depth_img,cv2.COLOR_BGR2RGB)
        self.lock2.acquire()
        output_img = cv2.addWeighted(self.image,0.6,depth_img,0.4,0)
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

