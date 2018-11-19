# -*- coding: utf-8 -*-
import tensorflow as tf


class generic_model():
    
    # Session  
    def __init__( self, 
                 model,
                 epochs,
                 save_name =None, 
                 upload_to_drive = True, 
                 generator = None, 
                 data = None ):

        self.model =model 
        self.epochs = epochs
        if save_name != None:
            self.save_name = save_name 
        self.generator = generator
        self.data = data 
        
        
        
    def __init__(forward_pass, 
                 loss, 
                 optmizer = 'adam' , 
                 epochs,
                 batch_size =32, 
                 save_name = , 
                 generator = None, 
                 data = None 
            ): 
        
    
        self.forward_pass =forward_pass
        self.loss = loss
        self.optimizer= optimizer 
        self.epochs = epochs
        self.batch_size = batch_size 
        self.upload_to_drive = upload_to_drive
        self.generator = generator
        self.data = data 
        
    
    
    @classmethod
    def train():
    
    
    
    