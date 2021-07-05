import os
import math
import random
import numpy as np

import rospy
import cv2
import message_filters  
import cv_bridge
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointField, CameraInfo
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs.point_cloud2
from sensor_msgs.point_cloud2 import read_points
from t import DeepLabNode
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from image_geometry import PinholeCameraModel
import yaml
from map import do_map, mini_name, map_kitti2mini

def create_pc_fields():
    fields = []
    fields.append( PointField( 'x', 0, PointField.FLOAT32, 1 ) )
    fields.append( PointField( 'y', 4, PointField.FLOAT32, 1 ) )
    fields.append( PointField( 'z', 8, PointField.FLOAT32, 1 ) )
    fields.append( PointField( 'i', 12, PointField.FLOAT32, 1 ) )
    fields.append( PointField( 'label', 16, PointField.UINT32, 1 ) )
    return fields

class CloudGenerator():
    def __init__(self, img_res = (672, 376)):
        self.camera_model = PinholeCameraModel()
        self.img_width, self.img_height = img_res
        self.isTransformSet = False
        self.isCamInfoSet = False
        self.seg_nn =DeepLabNode()
        self.num = 0
        self.load_extrinsics_ntu(fn='calib_data/velo-to-cam.txt', param= 'cam_to_velo')
        self.load_cam_info('calib_data/cam_param.yaml')
        self.build_cams_model()
        self.node = rospy.init_node('lidar_to_rgb', anonymous=True)
        self.bridge = CvBridge()
        self.pub_cloud = rospy.Publisher("/label_cloud", PointCloud2, queue_size = 1 )
        self.pub_image = rospy.Publisher("/projected_image",Image, queue_size = 1 )
        self.pub_velocity = rospy.Publisher("/odom", Odometry, queue_size=1)
        # self.sub_cam1_info = rospy.Subscriber(rospy.resolve_name( '/camera/left/camera_info'), CameraInfo, callback=self.cam1_info_callack, queue_size=1)
        self.sub_cam1_image = message_filters.Subscriber('/camera/left/image_raw', Image, queue_size=1)
        self.sub_cloud = message_filters.Subscriber('/velodyne_points', PointCloud2, queue_size=1)
        #self.sub_velocity = message_filters.Subscriber('/husky2_robot_pose', Odometry, queue_size=1)
        #self.sub_velocity = message_filters.Subscriber('/husky_velocity_controller/cmd_vel', Twist, queue_size=1)
        self.sub_velocity = message_filters.Subscriber('/odom_topic', Odometry, queue_size=1)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_cam1_image, self.sub_cloud, self.sub_velocity], 1, 0.08)
        self.ts.registerCallback(self.fusion_callback)
        
        rospy.spin()
        
    # def read_file(self, fn_vec, name, shape=(3,3), mode=''):
    #     content_vec = []
    #     for  file_name in fn_vec:
    #         for line in open(file_name, "r"):
    #             (key, val) = line.split(':', 1)
    #             if key == (name + mode):
    #                 content = np.fromstring(val, sep=' ').reshape(shape)
    #                 content_vec.append(content)
    #     return np.array(content_vec)
    def read_file(self, file_name, name, shape=(3,3), suffix='', sep=' '):
        content_vec = []

        for line in open(file_name, "r"):
            (key, val) = line.split(':', 1)
            if key == (name + suffix):
                content = np.fromstring(val, sep=sep).reshape(shape)
                content_vec.append(content)
        return np.array(content_vec)
    
    def load_cam_info(self, file_name):
        self.cam_info = CameraInfo()
        with open(file_name,'r') as cam_calib_file :
            cam_calib = yaml.load(cam_calib_file)
            self.cam_info.height = cam_calib['image_height']
            self.cam_info.width = cam_calib['image_width']
            self.img_width, self.img_height = self.cam_info.width, self.cam_info.height
            self.cam_info.K = cam_calib['camera_matrix']['data']
            self.cam_info.D = cam_calib['distortion_coefficients']['data']
            # self.cam_info.P = np.concatenate((self.cam_info.K, [0,0,1]), axis=1)
            # cam_info.P = cam_calib['projection_matrix']['data']
            self.cam_info.distortion_model = cam_calib['distortion_model']
            self.K = np.array(self.cam_info.K).reshape(3,3)
                
    def load_extrinsics_ntu(self, fn, param):
        def get_inverse(Mat):
            try:
                return np.linalg.inv(Mat)
            except np.linalg.LinAlgError:
                print("Failed to Inverse")
            #     # Not invertible. Skip this one.
                pass
        self.lidar_to_cams_RT_vect = get_inverse(self.read_file(fn, param, shape=(4,4), sep=',')) 
        self.isTransformSet = True
        
    def build_cams_model(self):
        self.cam_model=PinholeCameraModel()
        self.cam_model.fromCameraInfo(self.cam_info)
            

        self.isCamInfoSet = True
                   
    def load_extrinsics(self, fn_c2v_vec):
        
        self.cams_to_lidar_R_vec = [self.read_file(fn_c2v_vec, 'R')]
        self.cams_to_lidar_T_vec = [self.read_file(fn_c2v_vec, 'T', shape=(3,1))]
        to_homo = np.array([0, 0, 0, 1]).reshape(1, 4)
        self.cams_to_lidar_RT_vect = np.array([np.concatenate((np.concatenate((np.squeeze(R, axis=0), np.squeeze(T, axis=0)), axis=1), to_homo),axis=0) for R, T in zip(self.cams_to_lidar_R_vec, self.cams_to_lidar_T_vec)])
        # [print(np.squeeze(R, axis=0).shape, np.squeeze(T, axis=0).shape) for R, T in zip(self.cams_to_lidar_R_vec, self.cams_to_lidar_T_vec)]
        
        def get_inverse(Mat):
            try:
                return np.linalg.inv(Mat)
            except np.linalg.LinAlgError:
            #     # Not invertible. Skip this one.
                pass
            
        self.lidar_to_cams_RT_vect = [get_inverse(cam_tolidar_RT) for cam_tolidar_RT in self.cams_to_lidar_RT_vect]
        self.isTransformSet = True
        # print(self.lidar_to_cams_RT_vect[0] )

    # def cam1_info_callback(self, info_msg):
    #     self.img_width = info_msg.width
    #     self.img_height = info_msg.height
    #     self.camera_model.fromCameraInfo(info_msg)
    #     self.sub_cam1_info.unregister()
    #     self.isCamInfoSet = True
    #     print("cam1 resolution: ", (self.img_width, self.img_height))
    #     print("set cam1 info:", self.isCamInfoSet)
    #     print("cam1 tf:", self.camera_model.tfFrame())
        
    def fusion_callback(self, msg_img, msg_cloud, msg_velocity):
        os.makedirs('prepare_data/label/11', exist_ok=True)
        os.makedirs('prepare_data/img/11', exist_ok=True)
        os.makedirs('prepare_data/velocity/11', exist_ok=True)
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg_img, "bgr8")
            cv_image = cv2.resize(cv_image, (self.img_width, self.img_height))
            cv2.imwrite('prepare_data/img/11/%06d.png' % (self.num), cv_image)
        except CvBridgeError as e:
            print(e)
        try:
            velocity = []
            velocity.append([msg_velocity.twist.twist.linear.x, msg_velocity.twist.twist.linear.y, msg_velocity.twist.twist.linear.z, msg_velocity.twist.twist.angular.x, msg_velocity.twist.twist.angular.y, msg_velocity.twist.twist.angular.z])
            #velocity.append([msg_velocity.linear.x,msg_velocity.linear.y,msg_velocity.linear.z,msg_velocity.angular.x,msg_velocity.angular.y,msg_velocity.angular.z])
            np.savez('prepare_data/velocity/11/velocity%06d' % (self.num), vel=velocity)
        except:
            print('velocity error')
        label_mat = self.seg_nn.run_prediction(cv_image)
        label_mat = do_map(label_mat)
        if self.isTransformSet and self.isCamInfoSet:
            cv_temp = cv_image.copy();
            new_pts = []
            for point in (read_points(msg_cloud, skip_nans=True)):
                    pts_r = []
                    cam_index = 0
                    pts_xyz_homo = [point[0],point[1],point[2], 1.0]
                    intensity = point[3]
                    intensityInt = int(intensity*255)
                    
                    pts_xyz_cam = self.lidar_to_cams_RT_vect[cam_index].dot(pts_xyz_homo)
                    if pts_xyz_cam[2]<0 or pts_xyz_cam[2]>25:#0<depth<25
                        continue
                    pts_uvz_pix = self.K.dot((pts_xyz_cam[0], pts_xyz_cam[1], pts_xyz_cam[2]))
                    # pts_uv_pix = self.camera_model.project3dToPixel((pts_xyz_cam[0], pts_xyz_cam[1], pts_xyz_cam[2]))
                            # xy_n = xy_i / xy_i[2]
                    pts_uvz_pix = pts_uvz_pix/pts_uvz_pix[2]
                    pts_uv_pix = (pts_uvz_pix[0], pts_uvz_pix[1])
                    # print(pts_uv_pix)
                    #projection
                    if 0<=pts_uv_pix[0]<=self.img_width and 0<=pts_uv_pix[1]<=self.img_height:
                        cv2.circle(cv_temp, (int(pts_uv_pix[0]), int(pts_uv_pix[1])), 5, (intensityInt, intensityInt, intensityInt), thickness=-1 )
                        b,g,r = cv_image[int(pts_uv_pix[1]),int(pts_uv_pix[0])]
                        label = label_mat[int(pts_uv_pix[1]),int(pts_uv_pix[0])]
                        pts_r.append([point[0],point[1],point[2],point[3]])
                        new_pts.append([point[0],point[1],point[2],point[3],label])
            #np.savez('prepare_data/velody/%06d' % self.num, pointcloud=pts_r)
            np.savez('prepare_data/label/11/pts_l%06d' % (self.num), pts_l=new_pts)
            self.num += 1
            try:
                self.pub_image.publish(self.bridge.cv2_to_imgmsg(cv_temp, "bgr8"))
                ros_cloud = sensor_msgs.point_cloud2.create_cloud(msg_cloud.header, create_pc_fields(), new_pts)
                self.pub_cloud.publish(ros_cloud)
                self.pub_velocity.publish(msg_velocity)
            except CvBridgeError as e:
                print(e)
        else:
            print( 'Waiting for intrisincs and extrinsics')

if __name__ == '__main__':
    try:
        cloud_generator = CloudGenerator()
    except rospy.ROSInterruptException:
        pass