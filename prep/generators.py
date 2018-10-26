# -*- coding: utf-8 -*-

import pickle 
import numpy as np 

def validate_batch ():
    print('implement me' )


def one_hot():
    print('implement me now')

def coco_generator( mappings, captions, image_batch_szie = 1 ):

    
    acc_features = np.array([ ])
    acc_caption = np.array([ ])
    acc_target = np.array([ ])
    counter =0 
    for pkl_file, image_subset in mappings.items():
        
        with open(pkl_file, 'rb') as file :
            feature_dict = pickle.load( file )
        

        for image_name in image_subset:
                
            caption = captions[image_name]
            caption = np.array( [ [  line[:i]  for  i  in range(1,len(line)) ] for line in caption])

            target = np.array([ [ [ [i] for i in line[1:] ] for line in caption] ])
            target = [ i for i in target ]


            features = feature_dict [ image_name ]   
            features = features.repeat(3, axis=0)
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
                yield np.array( [features, caption]), target

            if len(acc_caption) != 0:
                yield [acc_features,acc_caption], acc_target
                
            

def create_coco_generator(mappings,
                          captions,
                          image_batch_size = 1):
    """
    inputs :
        DICT mapping : a dict keeping pkl to  list of images inside,
        DICT image_name_image_features 
        DICT captions: image_file_name to all captions_sequences 
        INT image_batch_size : number oF images every time theres a query
    return : 
        generator:
        total_steps:
    """

    ## Get the number of steps give the batch_size
    
    image_count = sum([len(i) for i in mappings.values()])
    steps = image_count//image_batch_size
    
    if int(image_count//image_batch_size)< image_count/image_batch_size:
        steps += 1
        generator = coco_generator( mappings, captions, image_batch_size  )

    return generator, steps

