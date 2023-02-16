#!usr/bin/env python3

import rospy
import tf2_ros
import pcl_ros
from sensor_msgs.msg import Image, PointCloud2
import ros_numpy
import numpy as np

class pc_project:
    def __init__(self):
        self.sub1 = rospy.Subscriber("/theia/right_camera/color/image_raw",Image,self.cam_callback)
        self.sub2 = rospy.Subscriber("/theia/os_cloud_node/points",PointCloud2,self.lidar_callback)
        self.image = None
        self.pcl_arr = None
        self.cam_intrinsic = np.array([[615.3355712890625, 0.0, 333.37738037109375], [0.0, 615.457763671875, 233.50408935546875], [0.0, 0.0, 1.0]])
        self.RT0 = np.array([[0, 1, 0 , 0], [-1, 0, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]],dtype=np.float32)
        self.RT1 = np.array([[ 1.0000000e+00,  1.2496948e-02,  3.3130646e-03, -8.8867188e-02],
                            [-1.2901306e-02,  9.4775391e-01,  3.1884766e-01,  9.0478516e-01],
                            [ 8.4686279e-04, -3.1884766e-01,  9.4775391e-01, -3.9111328e-01],
                            [ 0.,  0.,  0.,  1.]],dtype=np.float32)
        self.RT2 = np.array([[ 0.99316406, -0.0456543 , -0.10821533,  0.5678711 ],
                            [ 0.03170776,  0.99121094, -0.12731934,  0.265625  ],
                            [ 0.11309814,  0.12298584,  0.98583984, -0.4326172 ],
                            [ 0.        ,  0.        ,  0.        ,  1.        ]],dtype=np.float32)
        self.RT3 = np.array([[ 0.99853516, -0.04736328, -0.03225708,  0.01328278],
                            [ 0.04324341,  0.9921875 , -0.1184082 , -0.02172852],
                            [ 0.03762817,  0.11682129,  0.9926758 , -0.3659668 ],
                            [ 0.        ,  0.        ,  0.        ,  1.        ]],dtype=np.float32)
        self.RT4 = np.array([[ 1.        , -0.01420593, -0.0075531 , -0.02868652],
                            [ 0.01390839,  0.99902344, -0.03799438,  0.04470825],
                            [ 0.00808716,  0.03790283,  0.99902344, -0.11315918],
                            [ 0.        ,  0.        ,  0.        ,  1.        ]],dtype=np.float32)
        self.RT5 = np.array([[ 1.        , -0.00584793,  0.00870514, -0.0171051 ],
                            [ 0.0058403 ,  1.        ,  0.0010519 ,  0.03430176],
                            [-0.00870514, -0.00100136,  1.        ,  0.03448486],
                            [ 0.        ,  0.        ,  0.        ,  1.        ]],dtype=np.float32)

    def cam_callback(self,data):
        data = ros_numpy.numpify(data)
        
 
    def lidar_callback(self,data):
        self.pcl_arr = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data,remove_nans=True)
