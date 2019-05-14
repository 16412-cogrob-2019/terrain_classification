
import matplotlib.image as mpimg
import numpy as np
import os,sys
#import skimage.io as io
import cv2
        
    
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

def calculate_metrics(tp,tn,fp,fn):
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*precision*recall/(precision+recall)
    return accuracy, precision, recall, f1

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
    
                

def make_grayscale(images):
    black_pixel = np.zeros(3)
    white_pixel = 1
    for img in images:
        for i in range(len(img)):
            for j in range(len(img[i])):
                if(img[i][j].all() != black_pixel.all()):
                    img[i][j] = white_pixel
    images = np.asarray(images)
    return images

# #method to save predicted images
# def save_result(save_path,npyfile, name, grayscale = True):
#     #iterates over all images
#     for i,img in enumerate(npyfile):
#         #if grayscale we transform array
#         if(grayscale):
#             img = img[:,:,0]
#         #image is saved to specified path
#         io.imsave(os.path.join(save_path,name%(i+1)),img)
        
def load_image(infilename):
    img = cv2.imread(infilename)
    return convert_for_CNN(img)
    
def convert_for_CNN(img):
    img = cv2.resize(img,(256,256))
    #input must be between 0 and 1
    img = img/255.0
    #switch from BGR to RGB
    img = img[:,:,::-1]
    return img

        
def load_image_set(directory, name, f, t):
    test_images = [load_image(directory+name%i) for i in range(f,t+1)]
    return test_images
    
def fill_obstacles(obstacles):
    for obstacle in obstacles:
        obstacle.sort()
        for i, tup in enumerate(obstacle):
            if(i==0):
                prev_tuple=tup
            elif(prev_tuple[0]==tup[0] and prev_tuple[1]<tup[1]):
                    fill_pixel(obstacle,prev_tuple,i)
                    prev_tuple=(prev_tuple[0],prev_tuple[1]+1)
            else:
                prev_tuple=tup
    return obstacles        
    
def fill_pixel(obstacle,pixel,index):
    exists_above=False
    exists_below=False
    for tup in obstacle:
        if (tup[1]==pixel[1] and tup[0]<pixel[0]):
            exists_above=True
        if (tup[1]==pixel[1] and tup[0]>pixel[0]):
            exists_below=True
        if(exists_above and exists_below):
            obstacle.insert(index,pixel)
            return
        

def get_pixel_list_of_obstacles(segmented_image, threshold=0.22,generate_own_threshold=False,katie=False,min_length=100):
    list_of_obstacles=[]
    img_copy = segmented_image.copy()
    if generate_own_threshold:
        max=0
        for i,row in enumerate(segmented_image):
            for j,pixel in enumerate(row):
                if pixel>max:
                    max=pixel
        threshold=max/1.8
        threshold=threshold[0]
        if threshold<0.2:
            threshold=0.2
    print("threshold: ", threshold)
    if katie:
        threshold=int(255*threshold)
        img_copy=np.uint8(255*(img_copy))
        kernel = np.ones((3,3), np.uint8) 
        img_copy = cv2.dilate(img_copy, kernel, iterations=1)
        #img_copy=cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        #imgray = cv2.cvtColor(segmented_image,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(img_copy,threshold,255,0)
        #print(cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE))
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        pts = []
        total=0

        for i in range(len(contours)):
            total += 1
            img_copy = segmented_image.copy()
            cv2.drawContours(img_copy, contours, i, color=255, thickness=-1)
            #image_print(img)
            pts = np.transpose(np.where(np.any(img_copy == 255, axis=-1)))
            if(len(pts)>=min_length):
                list_of_obstacles.append(pts)
        return list_of_obstacles
    
    #if not katie
    for i,row in enumerate(img_copy):
        for j,pixel in enumerate(row):
            if(pixel>=threshold):
                img_copy[i][j]=0
                list_of_obstacles=explore_obstacle(img_copy,(i,j),list_of_obstacles,threshold,min_length)
    list_of_obstacles=fill_obstacles(list_of_obstacles)
    print("yann method is used")
    return list_of_obstacles

def explore_obstacle(segmented_image, coordinates, list_of_obstacles,threshold,min_length):
    coord_queue=[coordinates]
    list_of_coordinates=[]
    while (len(coord_queue)>0):
        i,j=coord_queue[0]
        find_vertical_points(segmented_image,coord_queue,-1,i,j,threshold) 
        find_vertical_points(segmented_image,coord_queue,1,i,j,threshold)
        check_nearby_point(segmented_image,coord_queue,i,j-1,threshold)
        check_nearby_point(segmented_image,coord_queue,i,j+1,threshold)
        list_of_coordinates.append((i,j))
        coord_queue.pop(0)
    if len(list_of_coordinates)>=min_length:
        list_of_obstacles.append(list_of_coordinates)
    return list_of_obstacles
 
def find_vertical_points(segmented_image, coord_queue, i_addition,i,j,threshold):
    for j_addition in range(-1,2):
        check_nearby_point(segmented_image,coord_queue,i+i_addition,j+j_addition,threshold)
        
def check_nearby_point(segmented_image,coord_queue,i,j,threshold):
    if (i>=len(segmented_image) or i<0 or j<0 or j>=len(segmented_image[i])):
        return
    if(segmented_image[i][j]>=threshold):
        coord_queue.append((i,j))
        segmented_image[i][j]=0