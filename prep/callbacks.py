# -*- coding: utf-8 -*-
try : 
    from google.colab import drive
except :
    print( 'this module is onlyo for colab')
from keras.callbacks import Callback


class reload_token(Callback):

    def __init__(self, method, args):
        
        self.method = method
        self.args = args
        
        super( reload_token, self)
        
    def on_epoch_end( self):
        try:
            self.method(self.args)
            print('\r', end ="")
        except :
            print( 'method not working trying to mount again auth it if asked')
            try: 
                drive.mount()
            except:
                print("module < drive > isn't imported, and mounting filed")
     
class save_to_drive(Callback):
    def __init__( self, spath):      
        self.savepath = spath 
        super(save_to_drive ,self)
    
    def on_epoch_end(self ):
        self.save(self.savepath)

class reset_gen(Callback):
    def __init__(self , iterator , gen, args):
        
        self.gen = gen
        self.args = args 
        self.iterator = iterator 
        
        super( reset_gen, self)
    def on_epoch_end(self):
        self.iterator = self.gen( self.args)