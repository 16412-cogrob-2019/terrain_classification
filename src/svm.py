import cv2
import numpy as np
from sklearn import svm
import pickle

import os

def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", cv2.resize(img,None, fx=.2,fy=.2))
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def get_data(a, b):
	# for img_number in [i for i in range(1,51) if i%a == b]:
	X = []
	y = []

	for img_number in [i for i in range(1,51) if i%a == b]:

		img_gt = cv2.imread("ground_truth/image" + str(img_number) + "_gt.png");
		img = cv2.imread("originals/image" + str(img_number) + ".png");
		# img_gt = np.array([[[38, 59, 218 ],
		# 					[0,0,0]]]).astype(np.uint8)
		# img = np.array([[[5, 0, 0 ],
		# 				[0,0, 0]]]).astype(np.uint8)

		rock_pixels = np.transpose(np.where(np.all(img_gt == [38,59,218], axis=-1)))
		#print('rock',rock_pixels)
		# image_print(img)
		pebble_pixels = np.transpose(np.where(np.all(img_gt == [83,214,129], axis=-1)))

		# print(rock_pixels)
		# for p in rock_pixels:
		# 	cv2.rectangle(img,tuple((p[1],p[0])),tuple((p[1],p[0])),(0,0,255),1)

		# for p in pebble_pixels:
		# 	cv2.rectangle(img,tuple((p[1],p[0])),tuple((p[1],p[0])),(0,255,0),1)

		# image_print(img)
		X_rock = [img[j][i] for j,i in rock_pixels]
		y_rock = ["Rock" for i in range(len(X_rock))]

		X_pebbles = [img[j][i] for j,i in pebble_pixels]
		y_pebbles = ["Pebbles" for i in range(len(X_pebbles))]

		#print('pebbles', X_pebbles)
		X = X+X_rock+X_pebbles
		y = y+y_rock+y_pebbles  
	#print('X', X,'y',y)  
	return X,y


def make_svm(X, y, kern='linear'):
	clf = svm.SVC(kernel=kern,max_iter=1000)
	clf.fit(X,y)
	return clf

def most_common(lst):
	''' lst is an np.array of predictions ('rock' or pebbles') for each pixel in obstacle '''
	freq_dict = {}
	for el in lst[0]:
		if el in freq_dict.keys():
			freq_dict[el] += 1
		else:
			freq_dict[el] = 1
	print(freq_dict)
	return max(freq_dict, key=lambda x: freq_dict[x])

def classify(img, clf, X):
	# print(X)
	classifications = []
	for obj in X:
		print('here')
		predictions = [clf.predict(color) for color in obj]
		# predictions = [clf.predict(img[j][i]) for i,j in obj]
		print(predictions)
		classifications.append(most_common(predictions))
	return classifications

def testing(clf, a, b):
	# for img_number in [i for i in range(1,51) if i%a != b]:
	X_obj= []

	for img_number in [1,1]:
		img_gt = cv2.imread("ground_truth/image" + str(img_number) + "_gt.png");
		img = cv2.imread("originals/image" + str(img_number) + ".png");

		obstacle_pixels = np.transpose(np.where(np.logical_or((img_gt==[38,59,218]).all(axis=-1),
																 (img_gt==[83,214,129]).all(axis=-1))))
		#print(obstacle_pixels)
		# im2,contours,hierarchy = cv2.findContours(img_dilation, 1, 2)
		# pts = []
		# for i in range(len(contours)):
		# 	cv2.drawContours(cimg, contours, i, color=255, thickness=-1)
		# 	pts = np.transpose(np.where(cimg == 255))

		X_obj.append([[img[j][i] for j,i in obstacle_pixels]])
	#print(X_obj)

	return classify(img, clf, X_obj)  



# X,y = get_data(25,0)
# print('step 1')
# clf = make_svm(X,y)
# print('step 2')
# clf_file = pickle.dumps(clf)
# print('step 3')


# f = open('clf.pkl', 'wb')
# # pickle.dump(clf, f)
# # f.close()

#f2 = open('/home/mers/catkin_ws/src/terrain_classification/src/clf.pkl', 'rb')
#clf = pickle.load(f2)

#print(testing(clf,25,1))
