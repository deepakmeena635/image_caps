# -*- coding: utf-8 -*-
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os 
from tqdm import tqdm 
import pickle

class no_urls_given(Exception):
    print( "no urls given\n give atleast urls and/or dir_urls")
    pass 



def prepare_vgg16():    
    print( 'downloading vgg 16 ')    
    from keras.applications.vgg16 import VGG16
    from keras import Model 
    model = VGG16()
    print( 'download complete')
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    return model


def chop_stuff( arr , size ):
    
    total_len = len(arr)
    parts = total_len//size    
    new_arr = [ arr[i*size : (i+1)*size ] for i in range(parts) ]
    if parts*size < total_len : 
        new_arr.append( arr[parts*size : ])
    
    return new_arr
    

def prepare_images(urls = [],
                    dir_url =None,
                    model = None, 
                    save_name = 'Features.pkl'):
    
    
    """
    model : keras model to use predict
    urls : list of complete paths to image files
    dir_url : should not contaon any sub folders evertytinh here should be files 
    """
    if model ==None  :
        model = prepare_vgg16()
    if ( len(urls) == 0 and dir_url == None ):
        raise no_urls_given
    
    if dir_url != None:
        remain = np.ravel([ j for i,j,k in os.walk(dir_url)])
        remain = [ dir_url+'/'+i for i in remain]
        urls.append(remain )
        urls = list(np.ravel(urls))


    features = dict()  
    print( "extracting_features")
    for path in tqdm(urls): 
        image = load_img( path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = path.split('/')[-1]
        features[image_id] = feature
    
    with open( save_name, 'wb') as fil :
        pickle.dump(obj = features, file = fil )
    
    return features

