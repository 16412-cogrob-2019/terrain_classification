# Terrain Classification Pipeline 

The main file for running our system is 'classification_node-3.py', which is the ROS node for our pipeline.

This file defines all of our subscripers and publishers.

Since images are published from the Raspicam faster than we can process them, we created a subscriber to the camera image topic that then only publishes images to a throttled_image topic when the previous image has been processed and published.  The actual pipeline subscribes to the throttled_image topic.

The pipeline is initiated whenever it recieves an image.  The image is first processed into a format the CNN is expecting and then passed into the CNN prediction method, predict_relevant.  This function outputs a list of lists, which contains lists of all the pixels in each obstacle in the image.  The SVM then converts each of these pixels into an RGB feature vector and outputs the classification for each obstacle that was identified by the CNN.  The function then iterates over all the identified pixels and colors them green or red based on the obstacle classiciation from the SVM.  This image is then published alongside the original to the classified_image topic.

The relavent code files to run this node include 'svm.py' which contains all of the functions for training and testing the SVM and 'test_methods.py' and 'predict_image.py' which contain all the CNN functions.  The CNN weights and a saved SVM classifier are also required.  All of these files are in the /scr folder of this directory.