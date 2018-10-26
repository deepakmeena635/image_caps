# -*- coding: utf-8 -*-
import pickle 
import numpy as np 
from keras.preprocessing.sequence import pad_sequences as pad

    
def one_hot( i, dict_size):

    a = np.zeros(dict_size )
    a[i] = 1
    return a 



def coco_generator( mappings, 
                   captions,        #used for one hotting the target should be vocabularies size                 
                   dict_size,
                   max_len,
                   image_batch_szie = 1,  
                   path_to_pkl_files = "." ,
                   pkl_file_extension = '.pkl'):
    
    acc_features = np.array([ ])
    acc_caption = np.array([ ])
    acc_target = np.array([ ])
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

            target = np.array([ [ [ [i] for i in line[1:] ] for line in caption] ])
            target = [ one_hot(i, dict_size) for i in target ]

            features = feature_dict[ image_name ]   
            features = (features.repeat( len(caption), axis=0 )).reshape(-1,1,4096)
            
            counter +=1 

            if image_batch_szie > 1 :
                acc_features = np.append(acc_features,features)
                acc_caption  = np.append(acc_caption,caption)
                acc_target   = np.append(acc_target,target)

                if counter == image_batch_szie :
                    yield [acc_features,acc_caption], acc_target
                    acc_features = np.array([ ])
                    acc_caption = np.array([ ])
                    acc_target = np.array([ ])
                    counter == 0 
            else :
                print("features.shape, caption.shape, tar shape ",  features.shape, caption.shape, target.shape)
                yield [[features, caption], target]

            if len(acc_caption) != 0:
                yield [acc_features,acc_caption], acc_target




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
