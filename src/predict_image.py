'''
This file contains one method that is called upon in the pipeline. The method further calls upon methods from test_methods.py.
It takes an image as input, and returns both the list of pixels that is predicted to contain obstacles, as well as the prediction of the image.
'''

from test_methods import *
from unet import *
import numpy as np

    
def predict_relevant(image):
    #make model
    model = get_unet()
    weights = 'weights.hdf5'
    #load premade weights
    model.load_weights(weights)
    #we make sure to be able to recreate the shape of previous image
    prev_shape = image.shape
    #image must be reshaped to be predicted upon
    image = np.reshape(image,(1,)+image.shape)
    #get predicted array
    result = model.predict(image)  
    #must reshape result from an array of images to a single image
    result = np.reshape(result,(prev_shape[0],prev_shape[1],1))
    #find relevant pixels 
    pixels = get_pixel_list_of_obstacles(result)
    
    return pixels, np.reshape(result,(prev_shape[0],prev_shape[1]))
