import cv2
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
from svm import classify, image_print
from predict_image import predict_relevant
import pickle
import copy
# import skimage.io as io
# from skimage.transform import resize

f2 = open('clf_10000_iters_5_0.pkl', 'rb')
clf = pickle.load(f2)

for img_number in [i for i in range(1,51) if i%5 == 0]:

	image = cv2.imread("originals/image" + str(img_number) + ".png");
	img = cv2.resize(image, (256,256))
	image = img.copy()
	print('img loaded')

	#the image is converted for a suitable representation that can be passed to the CNN
	rgb_img = convert_for_CNN(image)
	# get pixels from CNN -- list of obstacles, which are lists of pixels that make up each obstacle
	obstacles_lst = predict_relevant(rgb_img)
	print('CNN done')

	# turn list of pixels into list of feature vectorsfor each pixel in each obstacle
	X = []
	for obstacle_lst in obstacles_lst:
		X.append([[image[j][i]/255.0 for j,i in obstacle_lst]])


	# SVM classifications list - classifies each obstacle in the list
	classifications = classify(image, clf, X)
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

	image_print(vis)

	# uncomment this if you want to save output image
	# cv2.imwrite("output_comparison_image"+ str(img_number) + ".png", vis)
