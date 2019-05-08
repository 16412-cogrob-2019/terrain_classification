#! /usr/bin/env python


import rospy
import cv2
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError


class Classification:
    def __init__(self):
        rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.img_callback, queue_size = 1)
        self.image_pub = rospy.Publisher('/classified_image', Image, queue_size = 1)
        self.bridge = CvBridge()
        # import the CNN weights?
        # self.clf = (import trained svm)


    def img_callback(self, img):
        # convert ros image message into an open cv image
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        # grayscale:
        # gray_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # pass grayscale image into CNN weights
        # pixel_lst = weights(gray_im)

        # pass bounding pixels into SVM functions
        # predictions = classify(image, self.clf, pixel_lst)
        # 

        # creates a green box in upper left corner
        cv2.rectangle(image,(0,0),(100,100),(0,255,0),2)

        try:
            #print(self.bridge.cv2_to_imgmsg(image, "bgr8"))
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
        except CvBridgeError as e:
            print(e)
            return


if __name__ == "__main__":
    rospy.init_node("Classification")
    classifier = Classification()
    rospy.spin()