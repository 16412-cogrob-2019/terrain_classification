import rospy
import cv2
from sensor_msgs import Image
from cv_bridge import CvBridge, CvBridgeError
#import CNN weights
#import svm file
from svm import classify, most_common
from predict_image import predict_relevant
import pickle

class Classification:
	def __init__(self):
		rospy.Subscriber('/image', Image,self.img_callback, queue_size = 1)
		self.image_pub = rospy.Publisher('/classified_image', Image, queue_size = 1)
		self.bridge = CvBridge()
		self.clf = pickle.load(open('clf.pkl', 'rb')) #SVM classifier


	def img_callback(self, img):
		# convert ros image message into an open cv image
		try:
			image = self.bridge.imgmsg_to_cv2(img, "bgr8")
		except CvBridgeError as e:
			print(e)

		# compress
		cv2.resize(image, (256,256))

        # get pixels from CNN -- list of obstacles, which are lists of pixels that make up each obstacle
        obstacles_lst = predict_relevant(image)

        # turn list of pixels into list of feature vectorsfor each pixel in each obstacle
        X = []
        for obstacle_pixels in obstacles_lst:
        	X.append([[image[j][i] for i,j in obstacle_pixels]]) # double check whether its j,i or i,j
		
		# SVM classifications list - classifies each obstacle in the list
		classifications = classify(image, self.clf, X)

		# format the output, change the pixel values of obstacles to red or green based on classification
		for i in range(len(pixel_lst)):  # this should get index of obstacles, which should align with classifications
			pixel_lst[i]
			for p in pixel_lst[i]:
				if classifications[i] == 'Rock':
					cv2.rectangle(image,tuple((p[1],p[0])),tuple((p[1],p[0])),(0,0,255),1) #red pixel
				else:
					cv2.rectangle(image,tuple((p[1],p[0])),tuple((p[1],p[0])),(0,255,0),1) #green pixel


      	# Format the output to see original and classified image next to each other

		# Create an array big enough to hold both images next to each other.
		vis = np.zeros(256, 512), np.float32)

		mat1 = cv.CreateMat(256,256, cv.CV_32FC1)
		cv.Convert( img1, mat1 )

		mat2 = cv.CreateMat(256, 256, cv.CV_32FC1)
		cv.Convert( img2, mat2 )

		# Copy both images into the composite image.
		vis[:256, :256] = mat1
		vis[:256, 256:512] = mat2


		try:
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))
		except CvBridgeError as e:
			print(e)


if __name__ == "__main__":
	rospy.init_node("Classification")
	classifier = Classification()
	rospy.spin()