import cv2
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
from svm import classify, image_print
from predict_image import predict_relevant
import pickle
import copy
import skimage.io as io
from skimage.transform import resize

f2 = open('clf_10000_iters_5_0.pkl', 'rb')
clf = pickle.load(f2)

#for img_number in [i for i in range(1,51) if i%a != 0]:
for img_number in [1]:
	CNN_image = mpimg.imread("image" + str(img_number) + ".png", 0)
	# CNN_image = resize(CNN_image, (256, 256), anti_aliasing=True)
	# image_print(CNN_image)

	image = cv2.imread("originals/image" + str(img_number) + ".png");
	img = cv2.resize(image, (256,256))
	image = img.copy()
	print('img loaded')

	# rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# print(rgb_img.shape, type(rgb_img))
	# get pixels from CNN -- list of obstacles, which are lists of pixels that make up each obstacle
	obstacles_lst = predict_relevant(CNN_image/255)
	print('CNN done')

	# turn list of pixels into list of feature vectorsfor each pixel in each obstacle
	X = []
	for obstacle_lst in obstacles_lst:
		X.append([[image[j][i]/255.0 for i,j in obstacle_lst]]) # double check whether its j,i or i,j
	# print(obstacles_lst)

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
