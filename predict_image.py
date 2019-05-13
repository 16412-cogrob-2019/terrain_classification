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
    save_result("camera_resized",result,"image%d.png")

    result = np.reshape(result,(prev_shape[0],prev_shape[1],1))
    #find relevant pixels 
    pixels = get_pixel_list_of_obstacles(result)
    return pixels

def convert_for_CNN(img):
    #input must be between 0 and 1
    img = img/255
    #switch from BGR to RGB
    img = img[:,:,::-1]
    return img

