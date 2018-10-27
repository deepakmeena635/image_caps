# -*- coding: utf-8 -*-
import pickle 
import numpy as np 
from keras.preprocessing.sequence import pad_sequences as pad

    
def one_hot( i, dict_size):
    a = np.zeros(dict_size+1)
    a[i] = 1
    return a 



def coco_generator( mappings, 
                   captions,        #used for one hotting the target should be vocabularies size                 
                   dict_size,
                   max_len,
                   image_batch_szie = 1,  
                   path_to_pkl_files = "." ,
                   pkl_file_extension = '.pkl'):
    
    acc_features = np.array([[[ 0 for i in range(4096)]]])
    acc_caption = np.array([[ 0 for i in range(max_len) ]])
    acc_target = np.array([[  0 for i in range(dict_size+1)]])
    counter =0 
    
    
    for pkl_file, image_subset in mappings.items():
        
        with open(path_to_pkl_files+ '/' + pkl_file + pkl_file_extension , 'rb') as file :
            feature_dict = pickle.load( file )

        for image_name in image_subset:
            
            image_name = image_name.split('/')[-1]
            temp = []
            
            caption = captions[image_name]
            [  [ temp.append( line[:i] )  for i in range( 1,len(line) ) ]  for line in caption ] 
            caption = np.array( pad(  temp, maxlen = max_len, padding = 'post', value =0 ))
            
            temp = []
            [ np.array([ temp.append(i) for i in line[1:] ]) for line in captions[image_name] ] 
            temp = np.asarray( temp )
            target = np.array( [ np.array( one_hot(i, dict_size)) for i in temp ])
            
            features = feature_dict[ image_name ]   
            features = (features.repeat( len(caption), axis=0 )).reshape(-1,1,4096)
            
            counter +=1 

            if image_batch_szie > 1 :
                acc_features = np.append(acc_features,features, axis =0 )
                acc_caption  = np.append(acc_caption,caption, axis =0 )
                acc_target   = np.append(acc_target,target, axis =0 )

                if counter == image_batch_szie :
                    yield [acc_features[1:],acc_caption[1:]], acc_target[1:]
                    acc_features = np.array([[[ 0 for i in range(4096)]]])
                    acc_caption = np.array([[ 0 for i in range(max_len) ]])
                    acc_target = np.array([[  0 for i in range(dict_size+1)]])
                    counter =0
                continue
            else :
                yield [[features, caption], target]

    if len(acc_caption) > 1 :
        yield [acc_features[1:],acc_caption[1:]], acc_target[1:]




def create_coco_generator(mappings,
                          captions,
                          dict_size,
                          max_len, 
                          image_batch_size = 1):

    """
    inputs:        
        DICT mapping : a dict keeping pkl to  list of images inside,
        DICT image_name_image_features 
        DICT captions: image_file_name to all captions_sequences 
        INT image_batch_size : number oF images every time theres a query    
    return:
        generator:
        total_steps:
    """

    ## Get the number of steps give the batch_size
    image_count = sum([len(i) for i in mappings.values()])
    steps = image_count//image_batch_size
    
    if int(image_count//image_batch_size)< image_count/image_batch_size:
        steps += 1
    generator = coco_generator( mappings, captions, dict_size, max_len, image_batch_size)

    return generator, steps
