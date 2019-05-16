'''
This file contains methods that are used in the CNN step of our pipeline. 
Methods are either used to prepare an image for prediction in the CNN, or used to find clusters of points which constitutes obstacles. The x,y points of the pixels in these clusters can then be passed on to the SVM, in order to classify which type of obstable we are considering.
'''
import numpy as np
import cv2
        
#Inputs an image we want to perform prediction on
#Outputs thee same image in a format that is ready to enter the CNN
def convert_for_CNN(img):
    #makes sure that image is of correct size
    img = cv2.resize(img,(256,256))
    #input must be between 0 and 1
    img = img/255.0
    #switch from BGR to RGB
    img = img[:,:,::-1]
    return img
        
'''
get_pixel_list_of_obstacles
Inputs: segmented_image: grayscale image that is the output of the CNN
        threshold: the minimum probability of presence of an obstacle in a particular pixel that will be considered good enough         for considering it a part of an obstacle
        min_length:the minmum amount of pixels that an obstacle must have, this is to prevent the situations where the CNN             misclassifies random individual pixels
Output: list_of_obstacles: a list, where each element is a list corresponding to an obstacle. These inner lists are made up of         tuples with x,y coordinates of the pixels that constitute the obstacle
'''
def get_pixel_list_of_obstacles(segmented_image, threshold=0.15,min_length=100):
    list_of_obstacles=[]
    img_copy = segmented_image.copy()
    for i,row in enumerate(img_copy):
        for j,pixel in enumerate(row):
            #checks if the pixel corresponds to a new obstacle
            if(pixel>=threshold):
                #setting the pixel found as 0 to ensure we don't iterate through it again
                img_copy[i][j]=0
                #appends all other pixels that are part of this obstacle as a list of tuples
                list_of_obstacles=explore_obstacle(img_copy,(i,j),list_of_obstacles,threshold,min_length)
    #adds any pixels to the corresponding obstacle that are completely surrounded with pixels from the obstacle, but somehow did not get picked earlier
    list_of_obstacles=fill_obstacles(list_of_obstacles)
    return list_of_obstacles


'''
explore_obstacle
Inputs: segmented_image: grayscale image that is the output of the CNN
        coordinates: coordinates of a pixel that corresponds to a new obstacle found
        list_of obstacles: list of all previous obstacles found, represented as a list of list of tuples
        threshold: the minimum probability of presence of an obstacle in a particular pixel that will be considered good enough         for considering it a part of an obstacle
        min_length:the minmum amount of pixels that an obstacle must have, this is to prevent the situations where the CNN             misclassifies random individual pixels
Output: list_of_obstacles: an updated list of obstacles, including the one recently found
'''
def explore_obstacle(segmented_image, coordinates, list_of_obstacles,threshold,min_length):
    coord_queue=[coordinates]
    list_of_coordinates=[]
    #if the queue is empty we have found all pixels that are part of this obstacle
    while (len(coord_queue)>0):
        #takes the first coordinate of the queue and checks all nearby coordinates to see if they should be added as a part of the obstacle, and thus in the queue
        i,j=coord_queue[0] 
        #checks the three points below the current point, and adds them to queue if necessary
        find_vertical_points(segmented_image,coord_queue,-1,i,j,threshold) 
        #checks the three points above the current point, and adds them to queue if necessary
        find_vertical_points(segmented_image,coord_queue,1,i,j,threshold)
        #checks point to the left of current point, and adds it to queue if necessary
        check_nearby_point(segmented_image,coord_queue,i,j-1,threshold)
        #checks point to the right of current point, and adds it to queue if necessary
        check_nearby_point(segmented_image,coord_queue,i,j+1,threshold)
        list_of_coordinates.append((i,j))
        coord_queue.pop(0)
    #checks if this obstacle is large enough for it to be considered an obstacle
    if len(list_of_coordinates)>=min_length:
        list_of_obstacles.append(list_of_coordinates)
    return list_of_obstacles

'''
find_vertical_points
Inputs: segmented_image: grayscale image that is the output of the CNN
        coord_queue: current coordinates of pixels corresponding to a new obstacle found
        i_addition: the direction in the i-dimension of the new coordinates we are going to explore
        i: row of previous pixel
        j: column of previous pixel
        threshold: the minimum probability of presence of an obstacle in a particular pixel that will be considered good enough         for considering it a part of an obstacle
appends to queue any of the 3 adjacent pixels in the given i-direction that corresponds to an obstacle
'''
def find_vertical_points(segmented_image, coord_queue, i_addition,i,j,threshold):
    #goes through every of the 3 adjacent pixels in direction provided by i-addition
    for j_addition in range(-1,2):
        check_nearby_point(segmented_image,coord_queue,i+i_addition,j+j_addition,threshold)

'''
check_nearby_point
Inputs: segmented_image: grayscale image that is the output of the CNN
        coord_queue: current coordinates of pixels corresponding to a new obstacle found
        i: row of the new pixel we are now exploring
        j: row of the new pixel we are now exploring
        threshold: the minimum probability of presence of an obstacle in a particular pixel that will be considered good enough         for considering it a part of an obstacle
appends to queue any of the 3 adjacent pixels in the given i-direction
'''
def check_nearby_point(segmented_image,coord_queue,i,j,threshold):
    #makes sure we aren't moving outside of the dimentions of the image
    if (i>=len(segmented_image) or i<0 or j<0 or j>=len(segmented_image[i])):
        return
    #checks if the pixel has a high enough value to be included
    if(segmented_image[i][j]>=threshold):
        coord_queue.append((i,j))
        #sets the new pixel we included as 0 to make sure we won't go through it again
        segmented_image[i][j]=0
        
'''
fill_obstacles
Input: obstacles: a list, where each element is a list corresponding to an obstacle. These inner lists are made up of                  tuples with x,y coordinates of the pixels that constitute the obstacle

appends, for each obstacle, any pixel that is surrounded by an obstacle but didn't get included previously

It turns out that this method has a really bad computational complexity as the size of the obstacles grows, but this is only a problem in the case the CNN malfunctions and gives false positives to a very large portion of the image, as the obstacles otherwise have a size small enough for this to be perfectly negligible. This was not fixed in time, but now that we have good weights from our CNN this is no longer an issue as long as we don't try to classify something in a surrounding the CNN hasn't trained on
'''

def fill_obstacles(obstacles):
    for obstacle in obstacles:
        obstacle.sort()
        for i, tup in enumerate(obstacle):
            if(i==0):
                prev_tuple=tup
                #compares the current coordinates with the previous one, in our now sorted list, and sees if we find a gap within the row-dimension
            elif(prev_tuple[0]==tup[0] and prev_tuple[1]<tup[1]):
                    #checks if we find a gap in the column direction as well and adds the pixel in the current obstacle if that is the case
                    fill_pixel(obstacle,prev_tuple,i)
                    prev_tuple=(prev_tuple[0],prev_tuple[1]+1)
            else:
                prev_tuple=tup
    return obstacles        

'''
fill_pixel
Input: obstacle: a list of tuples corresponding to the coordinates of the pixels of a given obstacle
       pixel: the current pixel we are considering to fill in
       index: the index in the list where we would place this pixel if the conditions are sufficient, so that the list keeps          being sorted correctly
adds pixel to the list if it is surrounded, in the column-direction, by other pixels corresponding to the obstacle
       
'''
def fill_pixel(obstacle,pixel,index):
    exists_above=False
    exists_below=False
    for tup in obstacle:
        #checks if the tuple we have reached here is above the pixel
        if (tup[1]==pixel[1] and tup[0]<pixel[0]):
            exists_above=True
        #checks if the tuple we have reached here is below the pixel
        if (tup[1]==pixel[1] and tup[0]>pixel[0]):
            exists_below=True
        #adds pixel to the obstacle if it is surrounded in the column-dimension
        if(exists_above and exists_below):
            obstacle.insert(index,pixel)
            return
