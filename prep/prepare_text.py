import pickle
import numpy as np 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences as pad 
import json 
import os


def parse_json(path, 
               save_name_cap_pairs = True, 
               save_name =  'name_cap_pairs', rpwt = True  ):
    """
        returns : caption_id_dict: id to captions[] ,
        image_id_dict : id : file_name, 
        text : f_name '\t' desc \n * many (STRING )
        ==========
        rpwt : remove pairs having no '\t'
    """
    data = json.load( fp=open( path) )
    
    image_id_pairs  = [ [ i['file_name'], i['id']] for i in data['images']]
    image_id_dict = {j:i for i,j in image_id_pairs}
    caption_id_pairs = [[ i['caption'], i['image_id']] for i in data['annotations']]
    caption_id_dict = dict()

    for i,j in caption_id_pairs: 
        if j in caption_id_dict : 
            caption_id_dict[j].append(i)
        else: 
            caption_id_dict[j] = [i,]
            
    text = '\n'.join([ '\t'.join([ image_id_dict[j], i] ) for i, j in caption_id_pairs])
    
    if rpwt : 
        text = '\n'.join([i for i in text.split('\n') if '\t' in i])
        
    if save_name_cap_pairs:
        with open( save_name , 'w' ) as file: 
            file.write( text )
            
    return caption_id_dict, image_id_dict, text 

def load_doc(path):
	file = open(path, 'r')
	text = file.read()
	file.close()
	return text

def clean_text(text , isArray=True):
    """
        text : array of strings
    """
    def dPunct( line ):
        arr = [ i for i in line if (i.isalpha() or i==' '  or i== '\t') ]
        return ''.join(arr)
    
    if not isArray :
        text = text.split('\n')
    X = [ dPunct(i) for i in text ]
    if isArray: 
        return X
    else: 
        X = '\n'.join(X)
        return X
    

def process_text( text, to_pad = False, max_len = None, tok = None, save_name = None, num_word = None ):        
    """tips :
             tok is tokenizer:
                 pass tok as none while processing trainig data 
            text :
                should have line breaks and in each line there should be a '\t' 
                also it should be a String
                NAME : Caption formatted                 
        return :
            tok: tokenizer,
            text_dict:
    """
    names = [ i.split('\t')[0] for i in text.split('\n')]
    descs = [ i.split('\t')[1] for i in text.split('\n')]
    
    clean_descs = clean_text(descs)
    clean_descs = [ i.split(' ') for i in clean_descs ]
    
    if tok ==None :
        tok = Tokenizer( num_words= num_word )
        tok.fit_on_texts( clean_descs )
    desc_seqs = tok.texts_to_sequences(clean_descs)
    
    text_dict = dict( )
    
    if to_pad :
        if max_len== None  :
            max_len = max([len(i) for i in clean_descs])
        else:
            desc_seqs = pad(desc_seqs , maxlen= max_len, padding = 'post', value = 0  )
    
    for i in range(len(names)):    
        
        if names[i] in text_dict :
            (text_dict[names[i]]).append(desc_seqs[i])
        else :
            text_dict[names[i]] = [desc_seqs[i]]
            
    if not save_name ==None : 
        with open(save_name, 'wb') as fil :
            pickle.dump(obj = text_dict, file = fil)
        with open(save_name+'tokenizer', 'wb') as fil :
            pickle.dump(obj = tok, file = fil)

    return  tok, text_dict
