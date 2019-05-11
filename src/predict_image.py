from test_methods import *
from unet import *

    
def predict_relevant(image):
    #make model
    model = get_unet()
    weights = 'weights.hdf5'
    #load premade weights
    model.load_weights(weights)
    #we make sure to recreate shape of previous image
    prev_shape = image.shape
    #image must be reshaped to be predicted upon
    image = np.reshape(image,(1,)+image.shape)
    #get predicted array
    result = model.predict(image)
    #reshape back 
    result = np.reshape(result,(prev_shape[0],prev_shape[1],1))
    #find relevant pixels 
    pixels = get_pixel_list_of_obstacles(result)
    return pixels
