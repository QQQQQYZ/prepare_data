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


class CloudGenerator():
    def __init__(self, img_res=(672, 376)):
        self.node = rospy.init_node('republisher', anonymous=True)
        self.pub_odometry = rospy.Publisher("/odom_topic", Odometry, queue_size=1)
        # self.sub_cam1_info = rospy.Subscriber(rospy.resolve_name( '/camera/left/camera_info'), CameraInfo, callback=self.cam1_info_callack, queue_size=1)
        self.twist_subscriber = rospy.Subscriber('/husky_velocity_controller/cmd_vel', Twist, callback=self.twist_callback)
        self.odometry = Odometry()
    def twist_callback(self,data):
        self.odometry.header.stamp = rospy.Time.now()
        print(rospy.Time.now())
        self.odometry.twist.twist.linear.x = data.linear.x
        self.odometry.twist.twist.linear.y = data.linear.y
        self.odometry.twist.twist.linear.z = data.linear.z
        self.odometry.twist.twist.angular.x = data.angular.x
        self.odometry.twist.twist.angular.y = data.angular.y
        self.odometry.twist.twist.angular.z = data.angular.z
        self.pub_odometry.publish(self.odometry)


if __name__ == '__main__':
    try:
        cloud_generator = CloudGenerator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
