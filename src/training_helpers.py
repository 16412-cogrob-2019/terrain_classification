'''
This file contains methods that are used in order to evaluate the performance of a trained model. It includes methods to load and save images, as well as methods to calculate relevant performance metrics.
'''

from unet import *
from test_methods import *
import numpy as np
import os
import skimage.io as io
import cv2

#method used to calculate different metrics 
#to aseess performance of a proposed model
def calculate_metrics(tp,tn,fp,fn):
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*precision*recall/(precision+recall)
    return accuracy, precision, recall, f1

#method used to compare a predicted image to the ground truth in order to find true and false positives and negatives
#a treshold is included to force pixels to be 1 or 0, as most pixels are floating points
def confusion_matrix(truth, prediction,treshold):
    #make sure that every value is either 1 or 0
    p = prediction.copy()
    t = truth.copy()
    p[p > treshold ] = 1
    p[p <= treshold] = 0
    t[t > treshold ] = 1
    t[t <= treshold] = 0
    #convert arrays to lists
    t = list(t.flat)
    p = list(p.flat)
    #lists must be of equal length
    assert(len(t)==len(p))
    #initialize values
    tp = tn=fp=fn = 0
    #iterate in order to find true/false positive/negative
    for i in range(len(t)):
        if(t[i]==1 and p[i]==1):
            tp+=1
        elif(t[i]==0 and p[i]==0):
            tn+=1
        elif(t[i]==0 and p[i]==1):
            fp+=1
        elif(t[i]==1 and p[i]==0):
            fn+=1
    return tp,tn,fp,fn

#method is used to test newly trained weights, and return different scores 
def test_weights(weights,pred_dir, image_dir, image_name,gt_dir,gt_name, start,end,treshold):
    #load the weights
    model = get_unet()
    model.load_weights(weights)
    #load the testing images
    imgs = load_image_set(image_dir,image_name,start,end)
    imgs = np.asarray(imgs)
    #predict on the testing images
    prediction = model.predict(imgs)
    #load colored ground truth
    ground_truth = load_image_set(gt_dir,gt_name,start,end)
    #convert gt to grayscale
    grayscale_gt = make_grayscale(ground_truth)
    grayscale_gt = np.asarray(grayscale_gt)
    #save the predictions
    save_result(pred_dir,prediction, "prediction%d.png")
    #reshape arrays for the confusion matrix method
    grayscale_gt = grayscale_gt[:,:,:,0]
    prediction = prediction[:,:,:,0]
    #get confusion matrix
    tp,tn,fp,fn = confusion_matrix(grayscale_gt,prediction, treshold)
    #return accuracy, precision, recall and f1
    return calculate_metrics(tp,tn,fp,fn)
    
                
#method converts an image of multi-class ground truth image to grayscale binary class image
def make_grayscale(images):
    black_pixel = np.zeros(3)
    white_pixel = 1
    #iterates over all pixels in all images
    for img in images:
        for i in range(len(img)):
            for j in range(len(img[i])):
                #if pixel is not black, it is either green or red
                if(img[i][j].all() != black_pixel.all()):
                    #green or red pixels are set to white, to signify that they are both obstacles
                    img[i][j] = white_pixel
    #the images are returned as an array
    images = np.asarray(images)
    return images

#method to save predicted images
def save_result(save_path,npyfile, name, grayscale = True):
    #iterates over all images
    for i,img in enumerate(npyfile):
        #if the image is grayscale we transform array
        if(grayscale):
            img = img[:,:,0]
        #image is saved to specified path
        io.imsave(save_path+name%(i+1),img)
        
#method loads an image, and converts it so that it is ready to be used for prediction
def load_image(infilename):
    img = cv2.imread(infilename)
    return convert_for_CNN(img)

#loads a set of images
def load_image_set(directory, name, f, t):
    test_images = [load_image(directory+name%i) for i in range(f,t+1)]
    return test_images
    
