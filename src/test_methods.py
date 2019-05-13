
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
#import helpers as h
import os,sys
from os import *
#import skimage.io as io
#from PIL import Image



#method to load testing images
def testGenerator(test_path,general_name,f,t,target_size = (256,256)):
    for i in range(f,t+1):
        img = io.imread(os.path.join(test_path,general_name%i))
        img = img / 255
        img = np.reshape(img,(1,)+img.shape)
        yield img
        
#method to save predicted images
def save_result(save_path,npyfile):
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        io.imsave(os.path.join(save_path,"greyscale%d_gt.png"%(i+1)),img)
def load_image(infilename):
    return mpimg.imread(infilename)

def resize_image(filename, origin_dir, new_dir, new_size):
    img = Image.open(origin_dir+filename)
    img = img.resize(new_size,Image.NEAREST)
    img.save(new_dir+filename)
    
def get_pixel_list_of_obstacles(segmented_image, threshold=1,generate_own_threshold=True):
    if generate_own_threshold:
        max=0
        for i,row in enumerate(segmented_image):
            for j,pixel in enumerate(row):
                if pixel>max:
                    max=pixel
        threshold=max/1.8
        if threshold<0.2:
            threshold=0.2
    img=segmented_image.copy()
    list_of_obstacles=[]
    for i,row in enumerate(img):
        for j,pixel in enumerate(row):
            if(pixel>=threshold):
                img[i][j]=-1
                list_of_obstacles=explore_obstacle(img,(i,j),list_of_obstacles,threshold)
    return list_of_obstacles

def explore_obstacle(segmented_image, coordinates, list_of_obstacles,threshold):
    coord_queue=[coordinates]
    list_of_coordinates=[]
    index=0
    while (len(coord_queue)>0):
        index+=1
        i,j=coord_queue[0]
        if(i>0):
            if(segmented_image[i-1][j]>=threshold):
                coord_queue.append((i-1,j))
                segmented_image[i-1][j]=-1
            if(j>0 and segmented_image[i-1][j-1]>=threshold):
                coord_queue.append((i-1,j-1))
                segmented_image[i-1][j-1]=-1
            if(j+1<len(segmented_image[i]) and segmented_image[i-1][j+1]>=threshold):
                coord_queue.append((i-1,j+1))
                segmented_image[i-1][j+1]=-1
        if(i+1<len(segmented_image)):
            if(segmented_image[i+1][j]>=threshold):
                coord_queue.append((i+1,j))
                segmented_image[i+1][j]=-1
            if(j>0 and segmented_image[i+1][j-1]>=threshold):
                coord_queue.append((i+1,j-1))
                segmented_image[i+1][j-1]=-1
            if(j+1<len(segmented_image[i]) and segmented_image[i-1][j+1]>=threshold):
                coord_queue.append((i+1,j+1))
                segmented_image[i+1][j+1]=-1
        if(j>0):
            if(segmented_image[i][j-1]>=threshold):
                coord_queue.append((i,j-1))
                segmented_image[i][j-1]=-1
        if(j+1<len(segmented_image[i])):
            if(segmented_image[i][j+1]>=threshold):
                coord_queue.append((i,j+1))
                segmented_image[i][j+1]=-1
        list_of_coordinates.append((i,j))
        coord_queue.pop(0)
    if len(list_of_coordinates)>3:
        list_of_obstacles.append(list_of_coordinates)
    return list_of_obstacles
        
