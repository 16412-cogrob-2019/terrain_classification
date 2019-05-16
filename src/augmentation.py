'''
This file contains methods that are used to make generators that loads images from specific folders into the CNN. This is crucial when using augmentation, as we cannot store such large datasets in memory at once, so we need to load only a few at once.
'''

from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def adjustData(img,gt):
    #make sure img value is between 0 and 1, and that mask is either 1 or 0
    if(np.max(img) > 1):
        img = img / 255
        gt = gt /255
        gt[gt > 0.5] = 1
        gt[gt <= 0.5] = 0
    return (img,gt)

def trainGenerator(batch_size,path,image_folder,gt_folder,augmentation_variables,save_to_dir = None,
                    target_size = (256,256),seed = 1):

    #create generators generating coresponding images using same seed, that will yield coresponding images
    image_generator = ImageDataGenerator(**augmentation_variables)
    gt_generator = ImageDataGenerator(**augmentation_variables)

    image_generator = image_generator.flow_from_directory(
        path,
        classes = [image_folder],
        class_mode = None,
        color_mode = "rgb",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        seed = seed)

    gt_generator = gt_generator.flow_from_directory(
        path,
        classes = [gt_folder],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        seed = seed)

    #zip both generators into a shared generator
    train_generator = zip(image_generator, gt_generator)
    #adjust image values to be between 0 and 1, and gt images to be 0 or 1
    for (image,gt) in train_generator:
        image,gt = adjustData(image,gt)
        yield (image,gt)
