#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, cv2, cv_bridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np

BURGER_MAX_LIN_VEL = 0.22
BURGER_MAX_ANG_VEL = -2.84

WAFFLE_MAX_LIN_VEL = 0.26
#/camera/depth/image_rect_raw
WAFFLE_MAX_ANG_VEL = 1.82

LIN_VEL_STEP_SIZE = 0.005
ANG_VEL_STEP_SIZE = 0.1
control_linear_vel  = 0.0
target_linear_vel   = 0.1

class Tracking:
    def __init__(self):
      self.bridge = cv_bridge.CvBridge()
      self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
      self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
      self.twist = Twist()

    def constrain(self, input, low, high):
        if input > high:
          input = high
        elif input < low:
          input = low
        else:
          input = input

        return input

    def checkLinearLimitVelocity(self, vel):
        vel = self.constrain(vel, -BURGER_MAX_LIN_VEL, BURGER_MAX_LIN_VEL)
        return vel

    def image_callback(self, msg):
#       image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8') # opencv image use bgr8 -> ros image
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') # opencv image use bgr8 -> ros image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # change HSV
        
        lower_green = np.array([25, 43, 46])
        upper_green = np.array([46, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green) # cv2.inRange(img,low,high)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
#       cv2.imshow('mask', mask)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(image, center, 5, (0, 0, 255), -1)
            h, w, d = image.shape
            size = image.shape
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            err = cx - w / 2
            print( radius )
            
            if radius > 45  and radius < 160:
                global target_linear_vel
                self.twist.linear.x = 0.1
                target_linear_vel = 0.1
                self.twist.angular.z = -float(err) / 200
                self.cmd_vel_pub.publish(self.twist)
                
            elif radius > 160:
                global control_linear_vel
                #global target_linear_vel
                #target_linear_vel = target_linear_vel - LIN_VEL_STEP_SIZE
                target_linear_vel = self.checkLinearLimitVelocity(target_linear_vel - LIN_VEL_STEP_SIZE)
                
                if target_linear_vel <= 0:
                    self.twist.linear.x = 0.0
                    self.twist.angular.z = 0.0
                else:
                    print( target_linear_vel ) 
                    print( "---" ) 
                    self.twist.linear.x = target_linear_vel
                    #self.twist.angular.z = -float(err) / 500
                    
                print( 'go back' )
                self.cmd_vel_pub.publish(self.twist)
                
            elif radius < 45:
                print( "orange<45" )
                self.twist.linear.x = 0.0
                self.twist.angular.z =  -0.4
                self.cmd_vel_pub.publish(self.twist)
                
        elif len(cnts) <= 0:
            print( "no orange" )
            self.twist.linear.x = 0.0
            self.twist.angular.z = -0.4
            self.cmd_vel_pub.publish(self.twist)
               
        cv2.namedWindow("window2", cv2.WINDOW_NORMAL)
        cv2.imshow("window2", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.twist.linear.x = 0.0; #twist.linear.y = 0.0; twist.linear.z = 0.0
            self.twist.angular.z = 0.0
            
if __name__ == '__main__':
    try:
        rospy.init_node('track', anonymous=True)
        rospy.loginfo("Starting cv_bridge_test node")
        track = Tracking()
        rospy.spin()

    except KeyboardInterrupt:
        print ("Shutting down cv_bridge_test node.")
        cv2.destroyAllWindows()