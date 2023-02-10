#!usr/bin/env python3

import csv
import random
# import open3d as o3

import cv2
# import mathutils
# import matplotlib
# matplotlib.use('Qt5Agg')
import os
import matplotlib.pyplot as pltk
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel



from models.LCCNet import LCCNet

# from quaternion_distances import quaternion_distance
from lidar_camera_fusion.utils import (mat2xyzrpy, merge_inputs, overlay_imgs, quat2mat,
                   quaternion_from_matrix, rotate_back, rotate_forward,
                   tvector2mat)


from PIL import Image as im

from torchvision import transforms
import rospy
import tf2_ros
import pcl_ros
from sensor_msgs.msg import Image, PointCloud2
import ros_numpy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_2D_lidar_projection(pcl, cam_intrinsic):
    pcl_xyz = cam_intrinsic @ pcl.T
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
    pcl_uv = pcl_xyz[:, :2]

    return pcl_uv, pcl_z


def lidar_project_depth(pc_rotated, cam_calib, img_shape):
    pc_rotated = pc_rotated[:3, :].detach().cpu().numpy()
    cam_intrinsic = cam_calib
    pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated.T, cam_intrinsic)
    mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (
            pcl_uv[:, 1] < img_shape[0]) & (pcl_z > 0)
    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]
    pcl_uv = pcl_uv.astype(np.uint32)
    pcl_z = pcl_z.reshape(-1, 1)
    depth_img = np.zeros((img_shape[0], img_shape[1], 1))
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z
    depth_img = torch.from_numpy(depth_img.astype(np.float32))
    depth_img = depth_img.cuda()
    depth_img = depth_img.permute(2, 0, 1)
    pc_valid = pc_rotated.T[mask]

    return depth_img, pcl_uv, pc_valid

class lidar_cam:
    def __init__(self):
        tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(tfBuffer)
        self.sub1 = rospy.Subscriber("/theia/right_camera/color/image_raw",Image,self.cam_callback)
        self.sub2 = rospy.Subscriber("/theia/os_cloud_node/points",PointCloud2,self.lidar_callback)
        self.image = im.Image()
        self.pcl_arr = []
        self.cam_intrinsic = np.array([[615.3355712890625, 0.0, 333.37738037109375], [0.0, 615.457763671875, 233.50408935546875], [0.0, 0.0, 1.0]])
        self.to_tensor = transforms.ToTensor()
        self.normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        self.main()
    
    def cam_callback(self,data):
        data = ros_numpy.numpify(data)
        self.image = im.fromarray(data)
        self.sub1.unregister()
 
    def lidar_callback(self,data):
        self.pcl_arr = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data,remove_nans=True)
        self.pcl_arr = torch.from_numpy(self.pcl_arr.astype(np.float32))
        self.pcl_arr = self.pcl_arr.cuda()
        self.sub2.unregister()
    
    def main(self):
        input_size = (256, 512)
    
        weights = [
            '/home/cerlab/submodule_ws/src/lidar_camera_fusion/include/LCCNet/pretrained/kitti_iter1.tar',
            '/home/cerlab/submodule_ws/src/lidar_camera_fusion/include/LCCNet/pretrained/kitti_iter2.tar',
            '/home/cerlab/submodule_ws/src/lidar_camera_fusion/include/LCCNet/pretrained/kitti_iter3.tar',
            '/home/cerlab/submodule_ws/src/lidar_camera_fusion/include/LCCNet/pretrained/kitti_iter4.tar',
            '/home/cerlab/submodule_ws/src/lidar_camera_fusion/include/LCCNet/pretrained/kitti_iter5.tar'
        ]

        models = []

        for i in range(len(weights)):
            print(i)
            model = LCCNet(input_size, use_feat_from=1, md=4,
                                use_reflectance=False, dropout=0.0)
            checkpoint = torch.load(weights[i], map_location='cpu')
            saved_state_dict = checkpoint['state_dict']
            model.load_state_dict(saved_state_dict)
            model = model.to(device)
            model.eval()
            models.append(model)

        print('here')
        real_shape = np.array(self.image).shape
        real_shape = [real_shape[1],real_shape[2],real_shape[0]]
        depth_img,_,_ = lidar_project_depth(self.pcl_arr, self.cam_intrinsic, real_shape)
        depth_img = torch.unsqueeze(depth_img,0)
        # depth_img = depth_img.detach().cpu().numpy()
        # depth_img = (depth_img / np.max(depth_img)) * 255
        self.image.save('rgb.png')
        self.image = self.to_tensor(self.image)
        self.image = self.normalization(self.image)
        self.image = torch.unsqueeze(self.image,0)
        rgb_resize = F.interpolate(self.image, size=[256, 512], mode="bilinear")
        rgb_resize = rgb_resize.to(device)

        RTs = []
        
        for iteration in range(len(weights)):
            lidar_resize = F.interpolate(depth_img, size=[256, 512], mode="bilinear")
            lidar_resize = lidar_resize.to(device)
            T_predicted, R_predicted = models[iteration](rgb_resize, lidar_resize)
            RT_predicted = torch.mm(T_predicted, R_predicted)
            if iteration == 0:
                    rotated_point_cloud = self.pcl_arr
            rotated_point_cloud = rotate_forward(rotated_point_cloud, RT_predicted)
            RTs.append(RT_predicted)
            depth_img,_,_ = lidar_project_depth(rotated_point_cloud, self.cam_intrinsic, real_shape)
            depth_img = torch.unsqueeze(depth_img,dim=0)
            # depth_img = depth_img.detach().cpu().numpy()
            # depth_img = (depth_img / np.max(depth_img)) * 255
        
        depth_img.save('final_depth.png')
        
        
        print(RTs)






if __name__=='__main__':
    rospy.init_node('cam_lidar_transform')
    my_transformer = lidar_cam()
    
