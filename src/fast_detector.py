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
import matplotlib.pyplot as plt

class pc_project:
    def __init__(self):
        # self.sub1 = rospy.Subscriber("/theia/front_camera/color/image_raw",Image,self.cam_callback)
        self.sub1 = rospy.Subscriber("camera/color/image_raw",Image,self.cam_callback)
        self.sub2 = rospy.Subscriber("/theia/os_cloud_node/points",PointCloud2,self.lidar_callback)
        # self.sub2 = rospy.Subscriber("/pc_interpoled",PointCloud2,self.lidar_callback)
        self.sub3 = rospy.Subscriber("/theia/front_camera/aligned_depth_to_color/image_raw",Image,self.depth_cam_callback)
        self.sub3 = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw",Image,self.depth_cam_callback)
        self.sub4 = rospy.Subscriber("/theia/os_cloud_node/range_image",Image,self.depth_cam_callback2)
        self.pub1 = rospy.Publisher('/combined_output',Image,queue_size=1)
        self.image = None
        self.pcl_arr = None
        self.cam_intrinsic = np.array([[608.30517578125, 0.0, 333.8341064453125], [0.0, 608.1431884765625, 238.41688537597656], [0.0, 0.0, 1.0]])
        # self.cam_intrinsic = np.array([[910.1529541015625, 0.0, 636.0857543945312], [0.0, 910.1600952148438, 350.974365234375], [0.0, 0.0, 1.0]])
        self.RT0 = np.array([[0,-1,0,0], [0,0,-1, -0.435],[1,0,0, -0.324],[0, 0, 0, 1]],dtype=np.float32)
        # self.RT0 = np.array([[0,-1,0,0], [0,0,-1, 0.1],[1,0,0, -0.01],[0, 0, 0, 1]],dtype=np.float32)

        self.RTs = []
        for i in range(5):
            self.RTs.append(np.genfromtxt('./src/lidar_camera_fusion/cfg/RT_%d.csv'%i,delimiter=','))
        self.bridge = CvBridge()
        self.rate = rospy.Rate(30)
        self.lock = threading.Lock()
        self.lock2 = threading.Lock()
        self.lock3 = threading.Lock()
        self.orb = cv2.ORB_create(WTA_K=2)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def cam_callback(self,data):
        data = ros_numpy.numpify(data)
        self.lock2.acquire()
        # self.image = cv2.cvtColor(data,cv2.COLOR_BGR2RGB)
        self.image = data
        self.lock2.release()

    def depth_cam_callback(self,data):
        data = ros_numpy.numpify(data)
        data = ((data/np.max(data))*255).astype(np.uint8)
        data = cv2.applyColorMap(data,cv2.COLORMAP_JET)
        data = cv2.cvtColor(data,cv2.COLOR_BGR2RGB)
        self.lock3.acquire()
        self.depth_cam = data
        self.lock3.release()

    def depth_cam_callback2(self,data):
        data = ros_numpy.numpify(data)
        data = data[9:25,900:1200]
        data = ((data/np.max(data))*255).astype(np.uint8)
        data = cv2.resize(data,(640,480),interpolation=cv2.INTER_NEAREST)
        data = cv2.applyColorMap(data,cv2.COLORMAP_JET)
        data = cv2.cvtColor(data,cv2.COLOR_BGR2RGB)
        self.lidar_depth_cam = data

 
    def lidar_callback(self,data):
        data = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data,remove_nans=True)
        data = np.hstack((data,np.ones((data.shape[0],1))))
        data = data.T
        data = self.RT0 @ data
        # for RT in self.RTs:
        #     data = np.linalg.inv(RT) @ data

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
        
        self.lock3.acquire()
        cam_depth_img = self.depth_cam
        self.lock3.release()
        kp, des = self.orb.detectAndCompute(self.lidar_depth_cam,None)
        kp2, des2 = self.orb.detectAndCompute(cam_depth_img,None)
        matches = self.bf.match(des,des2)
        matches = sorted(matches, key = lambda x:x.distance)

        s = np.sum(depth_img, axis=2)
        non_black = (s != 0) 
        x, y = np.where(non_black)
        self.lock2.acquire()
        output_img = self.image
        self.lock2.release()
        output_img[x,y] = depth_img[x,y]

        img3 = cv2.drawMatches(self.lidar_depth_cam,kp,cam_depth_img,kp2,matches[:20],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.imshow(img3),plt.show(block=True)

        output_img = cv2.drawKeypoints(self.lidar_depth_cam,kp,None, color=(255,0,0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        rospy.loginfo("Output published")
        self.pub1.publish(self.bridge.cv2_to_imgmsg(img3,encoding="rgb8"))
        self.rate.sleep()
        

if __name__ == "__main__":
    rospy.init_node("depth_projector")
    my_obj = pc_project()
    while not rospy.is_shutdown():
        my_obj.project_depth()
    print("shutdown called")

