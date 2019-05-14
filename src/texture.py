from sklearn import svm
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import pickle
#from joblib import dump, load

X = [[1,1],[2,2]]
y = [0,1]
#clf = svm.SVC(gamma='scale')
#clf.fit(X, y)  
#s = pickle.dump(clf, open('save.pkl','wb'))
#clf = pickle.load(open('save.pkl','rb'))
#print(clf.predict([[3,3]]))
#print(z)


def get_nonblack(gt):
    vartype_uint8 = type(np.uint8(0))
    vartype_float32 = type(np.float32(0))
    
    if isinstance(gt[0,0],vartype_uint8):
        [ind1,ind2] = np.where(gt != 0)
        [ind_b1,ind_b2] = np.where(gt == 0)
    elif isinstance(gt[0,0],vartype_float32):
        [ind1,ind2] = np.where(gt != 0)
        [ind_b1,ind_b2] = np.where(gt == 0)
        print(gt[0,0].dtype)
    else:
        [ind1,ind2,z] = np.where(gt != [0,0,0])
        [ind_b1,ind_b2,z] = np.where(gt == [0,0,0])
    #print(len(ind1))
    #print(len(np.any(gt == [0, 0, 0], axis =-1)))
    #gt[ind1,ind2] = [255,255,255]
    return [[ind1, ind2], [ind_b1, ind_b2]]
    
def show(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def img_extract_obj(img,inds):
    vartype_uint8 = type(np.uint8(0))
    if isinstance(img[0,0],vartype_uint8):
        img[inds[0],inds[1]] = 0
    else:
        print(img.shape)
        img[inds[0],inds[1]] = [0,0,0]
    return img

def img_extract_obj_nan(img,inds):
    vartype_uint8 = type(np.uint8(0))
    if isinstance(img[0,0],vartype_uint8):
        img[inds[0],inds[1]] = np.nan
    else:
        img[inds[0],inds[1]] = [np.nan,np.nan,np.nan]
    return img

def crop_image(img,inds):
    img = img[np.min(inds[0]):np.max(inds[0]),np.min(inds[1]):np.max(inds[1])]
    #print(np.min(inds[0]),np.max(inds[0]),np.min(inds[1]),np.max(inds[1]))
    return img

def get_objs(img):
    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    [img_label,n_objs] = ndimage.label(img)
    obj_list = []
    for i in range(n_objs):
        [ind1,ind2,z] = np.where(img_label == i+1)
        [ind_b1,ind_b2,z] = np.where(img_label != i+1)
        obj_list.append([[ind1, ind2], [ind_b1, ind_b2]]) #want indices for obj 
    return obj_list

def get_objs_threshold(img, thres = .2):
    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(img)
    img = np.where(img > thres)
    [img_label,n_objs] = ndimage.label(img)
    obj_list = []
    print(img_label)
    for i in range(n_objs):
        print(np.where(img_label == i+1))
        [ind1,ind2] = np.where(img_label == i+1)
        [ind_b1,ind_b2] = np.where(img_label != i+1)
        obj_list.append([[ind1, ind2], [ind_b1, ind_b2]]) #want indices for obj 
    print('objlist: ',len(obj_list))
    return obj_list


def convert_objs(obj):
    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #[img_label,n_objs] = ndimage.label(img)
    print('obj size: ', len(obj))
    img_size = (256, 256)
    all_pixels = set([(i,j) for i in range(img_size[0]) for j in range(img_size[1])])
    obj_list = []
    for obstacle in obj:
        obstacle_pixels = set(obstacle)
        non_obstacle_pixels = all_pixels.difference(obstacle_pixels)
        non_obstacle_xs = []
        non_obstacle_ys = []
        for pixel in non_obstacle_pixels:
            non_obstacle_xs.append(pixel[0])
            non_obstacle_ys.append(pixel[1])
        obstacle_xs = []
        obstacle_ys = []
        for pixel in obstacle_pixels:
            obstacle_xs.append(pixel[0])
            obstacle_ys.append(pixel[1])
        obj_list.append([[obstacle_xs, obstacle_ys], [non_obstacle_xs, non_obstacle_ys]])
    print('len_obj_list: ',len(obj_list))
    # obj_list = []
    # for i in range(len(obj)):
    #   inds1 = np.zeros(len(obj[i]))
    #   inds2 = np.zeros(len(obj[i]))
    #   for j in range(len(obj[i])):
    #       inds1 = obj[i][0]
    #       inds2 = obj[i][1]

    #   obj_list.append([inds1,inds2],[])
    #   # [ind1,ind2,z] = np.where(img_label == i+1)
    #   # [ind_b1,ind_b2,z] = np.where(img_label != i+1)
    #   # obj_list.append([[ind1, ind2], [ind_b1, ind_b2]]) #want indices for obj 
    return obj_list

def features_hist(cleanedList):
    [y,x,z] = plt.hist(cleanedList,bins = 4)
    y_norm = y/np.sum(y)
    y_norm = np.ndarray.tolist(y_norm)
    #plt.show()
    #print(y)
    return y_norm   

def features_hist_abs(cleanedList):
    [y,x,z] = plt.hist(cleanedList,bins = 5)
    y_norm = y/np.sum(y)
    #y_dif = abs(np.max(y_norm)-y_norm)
    #y_dif[0] = [y_norm[2]
    y_dif = [0,0]
    y_dif[0] = y_norm[2]-y_norm[1]-y_norm[3]
    y_dif[1] = y_norm[2]-y_norm[0]-y_norm[4]
    #plt.show()
    #print(y)
    #y_dif  = np.array(y_dif)
    return y_dif

# def getFeature(img, gt, inds, inds_b):
def getFeature(img, inds, inds_b):
    
    edges = cv2.Canny(img,50,110) # best: >10,150
    #img_smooth = img 
    img_smooth = ndimage.gaussian_filter(img,sigma=1)
    img_smooth = img_extract_obj(img_smooth,inds_b)
    dx = ndimage.sobel(img_smooth,0)
    dy = ndimage.sobel(img_smooth,1)
    angle = np.arctan(np.divide(dy,dx))
    
    # remove background and non-edges
    [inds_edge, inds_b_edge] = get_nonblack(edges)
    angle = img_extract_obj_nan(angle,inds_b_edge)
    angle = img_extract_obj_nan(angle,inds_b)

    # convert from ndarray to flat list
    angle_list_cropped = np.ndarray.tolist(angle)
    angle_list_cropped_flat = [None] * len(angle_list_cropped)*len(angle_list_cropped[0])#*len(angle_list_cropped[0][0])
    k=0
    for i in range(len(angle_list_cropped)):
        for j in range(len(angle_list_cropped[i])):
            angle_list_cropped_flat[k] = angle_list_cropped[i][j][1]
            #angle_list_cropped_flat[k+1] = angle_list_cropped[i][j][1]
            #angle_list_cropped_flat[k+2] = angle_list_cropped[i][j][2]
            k=k+1#3
    
    # remove NaNs
    cleanedList = [x for x in angle_list_cropped_flat if str(x) != 'nan']
    features = features_hist_abs(cleanedList);
    #show(angle)
    return features
    
    #show(edges)

    # reshape into feature vector

def extract_features_img(gt,im,vectors_x):
    # get indices for object pixels
    #obj_list = get_objs(gt)
    obj_list = convert_objs(gt)
    
    for i in range(len(obj_list)):
        [inds, inds_b] = obj_list[i]

        #gt_cropped = crop_image(gt,inds) #get cropped image
        #im_cropped = crop_image(im,inds) #get cropped image
        #[inds, inds_b] = get_nonblack(gt_cropped)
        
        #y = getFeature(im_cropped, gt_cropped, inds, inds_b)
        #y = getFeature(im_cropped, inds, inds_b)
        y = getFeature(im, inds, inds_b)
        print(y)
        vectors_x.append(y)
    return vectors_x    

def extract_labeled_features_img(gt,im,vectors_training,class_training):
    # get indices for object pixels
    obj_list = get_objs(gt)
    
    for i in range(len(obj_list)):
        [inds, inds_b] = obj_list[i]

        # crop images
        gt_cropped = crop_image(gt,inds) #get cropped image
        im_cropped = crop_image(im,inds) #get cropped image
        [inds, inds_b] = get_nonblack(gt_cropped)
        
        # get features
        y = getFeature(im_cropped, gt_cropped, inds, inds_b)
        print(y)
        vectors_training.append(y)

        # get correct classes from truth images
        index_to_check = int(np.floor(len(inds[0])/2));
        r_val = gt_cropped[inds[0][index_to_check],inds[1][index_to_check]][2]
        g_val = gt_cropped[inds[0][index_to_check],inds[1][index_to_check]][1]
        if r_val > g_val:
            class_val = "Rock" # rock 0
        else:
            class_val = "Pebbles" # pebbles 1
        class_training.append(class_val)
        #print(class_val)
        #show(im_cropped)

    return vectors_training, class_training

def predict_set(clf,n_pred_start,n_pred_end):

    # PREDICT
    vectors_pred = []
    class_pred = []
    pred_list = []
    correct_list  = []
    for i in range(n_pred_start,n_pred_end+1):
        fn_num = i
        gt_fn = 'images/image' + str(fn_num) + '_gt.png'
        gt = cv2.imread(gt_fn)
        im_fn = 'images/image' + str(fn_num) + '.png'
        im = cv2.imread(im_fn)
        
        # gt = cv2.resize(gt, (256,256)) # simulate cnn input
        new_size1 = 640
        new_size2 = 480
        im = cv2.resize(im, (new_size1,new_size2))
        gt = cv2.resize(gt, (new_size1,new_size2))

        vectors_pred, class_pred = extract_labeled_features_img(gt,im,vectors_pred,class_pred)

    prediction = clf.predict(vectors_pred)
    return prediction, vectors_pred, class_pred

def predict_img(clf,im,im_seg):

    # PREDICT
    vectors_pred = []
    
    vectors_pred = extract_features_img(im_seg,im,vectors_pred)

    prediction = clf.predict(vectors_pred)
    print('prediction: ', prediction)
    return prediction#, vectors_pred


def show_results(prediction,vectors_pred,class_pred,vectors_training = []):
    correct_list  = []      
    pred_list = []  
    for i in range(len(prediction)):
        print(prediction[i],class_pred[i])
        if prediction[i] == class_pred[i]:
            correct_list.append(1)
        else:   
            correct_list.append(0)
        if prediction[i] == "Rock":
            pred_list.append("Rock")
        else:
            pred_list.append("Pebbles")

    print('answers: ', pred_list)
    print('correct?: ', correct_list)
    print('Accuracy: ', np.mean(correct_list))

    # compile classes
    class0x = []
    class0y = []
    class1x = []
    class1y = []
    class0x_pc = []
    class0y_pc = []
    class1x_pc = []
    class1y_pc = []
    class0x_pi = []
    class0y_pi = []
    class1x_pi = []
    class1y_pi = []
    
    for i in range(len(vectors_training)):
        if class_training[i] == "Rock":
            class0x.append(vectors_training[i][0])
            class0y.append(vectors_training[i][1])
        else:
            class1x.append(vectors_training[i][0])
            class1y.append(vectors_training[i][1])

    for i in range(len(vectors_pred)):
        if pred_list[i] == "Rock":
            if correct_list[i] == 1:
                class0x_pc.append(vectors_pred[i][0])
                class0y_pc.append(vectors_pred[i][1])
            else:
                class0x_pi.append(vectors_pred[i][0])
                class0y_pi.append(vectors_pred[i][1])
        else:
            if correct_list[i] == 1:
                class1x_pc.append(vectors_pred[i][0])
                class1y_pc.append(vectors_pred[i][1])
            else:
                class1x_pi.append(vectors_pred[i][0])
                class1y_pi.append(vectors_pred[i][1])
    plt.show()
    plt.plot(class0x,class0y,'ro') # rocks
    plt.plot(class1x,class1y,'go') # rocks
    plt.plot(class0x_pc,class0y_pc,'rD') # rocks
    plt.plot(class1x_pc,class1y_pc,'gD') # rocks
    plt.plot(class0x_pi,class0y_pi,'rx') # rocks
    plt.plot(class1x_pi,class1y_pi,'gx') # rocks
    plt.show()



## MAIN

# TRAINING
# trainflag = False;
# pkl_fn = 'texture_svm.pkl' #'save_t20_comp256_test.pkl'


# vectors_training = []

# if trainflag:
#   n_train = 20

#   # prep training vectors
#   vectors_training = []
#   class_training = []

#   for i in range(1,n_train+1):
#       # process image
#       fn_num = i
#       gt_fn = 'images/image' + str(fn_num) + '_gt.png'
#       gt = cv2.imread(gt_fn)
#       im_fn = 'images/image' + str(fn_num) + '.png'
#       im = cv2.imread(im_fn)
#       # im = cv2.resize(im, (256,256))
#       # gt = cv2.resize(gt, (256,256))

#       vectors_training, class_training = extract_labeled_features_img(gt,im,vectors_training,class_training)
#   print(class_training)
#   # train svm
#   X = vectors_training
#   y = class_training
#   clf = svm.SVC(gamma='scale')
#   clf.fit(X, y)  
    
#   #pickle
#   s = pickle.dump(clf, open(pkl_fn,'wb'))
    


# clf = pickle.load(open(pkl_fn,'rb'))
# n_pred_start = 37
# n_pred_end = 39
# [prediction, vectors_pred, class_pred] = predict_set(clf,n_pred_start,n_pred_end)
# show_results(prediction,vectors_pred,class_pred,vectors_training)



# test_num = 38
# im_fn = 'images/image' + str(test_num) + '.png'
# im = cv2.imread(im_fn)
# im_seg_fn = 'images/image' + str(test_num) + '_gt.png'
# im_seg = cv2.imread(im_seg_fn)

# clf = pickle.load(open(pkl_fn,'rb'))
# prediction = predict_img(clf,im,im_seg)
# print(prediction)


# [prediction, vectors_pred] = predict_img(clf,im,im_seg)