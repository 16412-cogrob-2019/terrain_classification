#! /usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
#import CNN weights
#import svm file
from svm import classify, most_common
from predict_image import predict_relevant
import pickle
#import matplotlib.image as mpimg
#from PIL import Image
import numpy as np
from svm import classify, image_print
from predict_image import predict_relevant
import copy

class Classification:
    def __init__(self):
        rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage ,self.img_callback, queue_size = 1)
        self.image_pub = rospy.Publisher('/classified_image', Image, queue_size = 1)
        self.bridge = CvBridge()
        self.clf = pickle.load(open('clf_p2.pkl', 'rb')) #SVM classifier
	self.i = True

    def img_callback(self, img):
        # convert ros image message into an open cv image
        if self.i:
            self.i = False
	    return
	try:
            image = self.bridge.compressed_imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
            print(e)

    
        img = cv2.resize(image, (256,256))
        image = img.copy()
        print('img loaded')

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # get pixels from CNN -- list of obstacles, which are lists of pixels that make up each obstacle
        obstacles_lst = predict_relevant(rgb_img/255)
        print('CNN done')

        # turn list of pixels into list of feature vectorsfor each pixel in each obstacle
        X = []
        for obstacle_lst in obstacles_lst:
            X.append([[image[j][i]/255.0 for j,i in obstacle_lst]])


        # SVM classifications list - classifies each obstacle in the list
        classifications = classify(image, self.clf, X)
        print('svm done')

        # format the output, change the pixel values of obstacles to red or green based on classification
        for i in range(len(obstacles_lst)):  # this should get index of obstacles, which should align with classifications
            pixel_lst = obstacles_lst[i]
            for p in pixel_lst:
                if classifications[i] == 'Rock':
                    cv2.rectangle(image,tuple((p[1],p[0])),tuple((p[1],p[0])),(0,0,255),1) #red pixel
                else:
                    cv2.rectangle(image,tuple((p[1],p[0])),tuple((p[1],p[0])),(0,255,0),1) #green pixel


        # Format the output to see original and classified image next to each other
        vis = np.concatenate((img, image), axis=1)

        #image_print(vis)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))
        except CvBridgeError as e:
            print(e)


if __name__ == "__main__":
    rospy.init_node("Classification")
    classifier = Classification()
    rospy.spin()