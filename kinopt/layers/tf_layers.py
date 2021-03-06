'''Layers which use tensorflow, so requires tf backend.
    Layers mostly directly using tensorflow's image tools

'''

import tensorflow as tf
import keras.backend as K
from keras.layers import Layer
import numpy as np

class Jitter2D(Layer):
    def __init__(self,
             jitter=16,
             **kwargs):
        super(Jitter2D, self).__init__(**kwargs)
        if isinstance(jitter,int):
            self.jitter = (jitter,jitter)
        else:
            assert isinstance(jitter,(tuple,list)) and len(jitter)==2
            self.jitter = jitter
            
    def call(self,x):
        if K.image_data_format() == 'channels_first':
            x = tf.transpose(x,[0,2,3,1])
            
        _,h,w,_ =  x.get_shape().as_list()
        x = tf.image.resize_image_with_crop_or_pad(x,h+self.jitter[0],
                             w+self.jitter[1])
        y_shift_1 = tf.random_uniform([],minval=0,
                                    maxval=(self.jitter[0]),
                                    dtype=tf.int32)
        x_shift_1 = tf.random_uniform([],minval=0,
                                    maxval=(self.jitter[1]),
                                    dtype=tf.int32)
        x = tf.image.crop_to_bounding_box(x,y_shift_1,x_shift_1,h,w)
        
        if K.image_data_format() == 'channels_first':
            x = tf.transpose(x,[0,3,1,2])
        
        return x
    
    def get_config(self):
        config = {'jitter': self.jitter}
        base_config = super(Jitter2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class RandomCrop2D(Layer):
    '''Should work like Jitter2D, but uses the random_crop function
        TODO: Figure out whats wrong, errors under strange circumstances
    '''
    def __init__(self,
             jitter=16,
             **kwargs):
        super(RandomCrop2D, self).__init__(**kwargs)
        if isinstance(jitter,int):
            self.jitter = (jitter,jitter)
        else:
            assert isinstance(jitter,(tuple,list)) and len(jitter)==2
            self.jitter = jitter

    def call(self,x):
        if K.image_data_format() == 'channels_first':
            nb,nch,h,w =  x.get_shape().as_list()
            new_shape =(nb,nch,h-self.jitter[0],w-self.jitter[1])

        elif K.image_data_format() == 'channels_last':
            nb,h,w,nch =  x.get_shape().as_list()
            new_shape =(nb,h-self.jitter[0],w-self.jitter[1],nch)
            
        x =  tf.random_crop(x,new_shape)
        x.set_shape(new_shape)
        return x
    
    def get_config(self):
        config = {'jitter': self.jitter}
        base_config = super(RandomCrop2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Crop2D(Layer):
    def __init__(self,
            offset_height,
            offset_width,
            target_height,
            target_width,
             **kwargs):
        super(Crop2D, self).__init__(**kwargs)
        self.offset_height = offset_height
        self.offset_width = offset_width
        self.target_height = target_height
        self.target_width = target_width

    def call(self,x):
        if K.image_data_format() == 'channels_first':
            x = tf.transpose(x,perm=[0,2,3,1])
        x = tf.image.crop_to_bounding_box(x,
                                          offset_height=self.offset_height,
                                          offset_width=self.offset_width,
                                          target_height=self.target_height,
                                          target_width=self.target_width
                                          )
        if K.image_data_format() == 'channels_first':
            x = tf.transpose(x,perm=[0,3,1,2])

        return x
    def compute_output_shape(self,input_shape):
        assert len(input_shape) == 4
        if K.image_data_format() == 'channels_first':
            nb,nch,h,w = input_shape
            return (nb,nch,self.target_height,self.target_width)
        elif K.image_data_format() == 'channels_last':
            nb,h,w,nch = input_shape
            return (nb,self.target_height,self.target_width,nch)
    
    def get_config(self):
        config = {'target_height':self.target_height,
                  'target_width':self.target_width,
                  'offset_height':self.offset_height,
                  'offset_width':self.offset_width}
        base_config = super(Crop2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Resize2D(Layer):
    def __init__(self,
                 new_h,
                 new_w,
                 **kwargs):
        super(Resize2D, self).__init__(**kwargs)
        self.h = new_h
        self.w = new_w

    
    def call(self,x):
        x_shape = x.get_shape().as_list()
        
        if K.image_data_format() == 'channels_first':
            x = tf.transpose(x,[0,2,3,1])
            
        x = tf.image.resize_bilinear(x,(self.h,self.w))
        
        if K.image_data_format() == 'channels_first':
            x = tf.transpose(x,[0,3,1,2])
        x.set_shape(self.compute_output_shape(x_shape))
        return x
    
    def compute_output_shape(self,input_shape):
        assert len(input_shape) == 4
        if K.image_data_format() == 'channels_first':
            nb,nch,h,w = input_shape
            return (nb,nch,self.h,self.w)
        elif K.image_data_format() == 'channels_last':
            nb,h,w,nch = input_shape
            return (nb,self.h,self.w,nch)
        
    def get_config(self):
        config = {'new_h': self.h,'new_w': self.w}
        base_config = super(Resize2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
#TODO: Force back shape?
class RandomResize2D(Layer):
    def __init__(self,
                 resize_vals=None,
                 **kwargs):
        super(RandomResize2D, self).__init__(**kwargs)
        if resize_vals is None:
                resize_vals = [0.95,0.975,1.,1.025,1.05]
        self.resize_vals = resize_vals
        self.logs = [1. for _ in self.resize_vals]
        self.resize_vals_tf =  tf.convert_to_tensor(self.resize_vals)

    
    def call(self,x):
        x_shape = x.get_shape()
        _,h,w,_ =  x_shape.as_list()

        sample = tf.multinomial(tf.log([self.logs]), 1)
        resize = self.resize_vals_tf[tf.cast(sample[0][0], tf.int32)]
        new_size = tf.cast(tf.stack((resize*h,resize*w),axis=0),dtype=tf.int32)
        x = tf.image.resize_bilinear(x,new_size)
        x = tf.image.resize_image_with_crop_or_pad(x,h,w)
        return x
    
    def get_config(self):
        config = {'resize_vals': self.resize_vals}
        base_config = super(RandomResize2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
class RandomRotate2D(Layer):
    def __init__(self,
             rotate_vals=None,
             angle_mode='degrees',
             **kwargs):
        super(RandomRotate2D, self).__init__(**kwargs)
        self.angle_mode = angle_mode
        if rotate_vals is None:
                rotate_vals = [-5.,-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.,5.]
        self.rotate_vals = rotate_vals
        
        if self.angle_mode == 'degrees':
            rad_vals = [np.pi*deg/180. for deg in self.rotate_vals]
            
        self.logs = [1. for _ in self.rotate_vals]
        self.rotate_vals_tf =  tf.convert_to_tensor(rad_vals)

    def call(self,x):
        rotate_sample = tf.multinomial(tf.log([self.logs]), 1)
        rotate = self.rotate_vals_tf[tf.cast(rotate_sample[0][0], tf.int32)]
        x = tf.contrib.image.rotate(x,rotate)
        return x

    def get_config(self):
        config = {'rotate_vals': self.rotate_vals,
                  'angle_mode':self.angle_mode}
        base_config = super(RandomRotate2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ChannelDecorrelate(Layer):
    def __init__(self,whitening_matrix=None,**kwargs):
        super(ChannelDecorrelate, self).__init__(**kwargs)
        if whitening_matrix is None:

            wht_mat = np.array(
                               [[0.6317 , 0.5744, 0.5204],
                               [0.      , 0.2557, 0.2949],
                               [0.      , 0.    , 0.2675]],
                                        dtype=np.float32)
        elif type(whitening_matrix) is np.ndarray:
            wht_mat = np.float32(whitening_matrix)
            
        elif whitening_matrix == 'lucid':
            '''Whitening matrix used by the Lucid library, 
                Uses PCA whitening (?)
            '''
            wht_mat = np.array(
                            [[0.5628,  0.5844,  0.5844],
                            [ 0.1948,  0.,     -0.1948],
                            [ 0.0432, -0.1082,  0.0649]],
                                        dtype=np.float32)
        else:
            #TODO: Check if tensorflow tensor
            raise Exception('whitening_matrix must be a numpy array')
            
        self.whitening_matrix = wht_mat
        
        '''
        Some other whitening matrices (computed from same data, 
        but different transforms of the input)
       
        whitening_matrix = np.array(
                            [[ 0.69512309, -0.71428814,  0.08121785],
                            [ 0.        ,  0.72797713, -0.56444855],
                            [ 0.        ,  0.        ,  0.3145091 ]],
                        dtype=np.float32)
        '''
        
    def call(self,x):
        x_shape = x.get_shape()
        
        if K.image_data_format() == 'channels_last':
            x = tf.tensordot(x,self.whitening_matrix,axes=[[3],[0]])
        elif K.image_data_format() == 'channels_first':
            x = tf.tensordot(self.whitening_matrix,x,axes=[[1],[0]])
        else:
            raise ValueError('unknown K image_data_format')
        x.set_shape(x_shape)
        return x
    
    def get_config(self):
        config = {'whitening_matrix':self.whitening_matrix}
        base_config = super(ChannelDecorrelate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

#Fourier layers require input shape (x,y) being known, so batch_input_shape
#must be given. kinopt.models.load_model forces this.
    
class FourierScaling(Layer):
    
    def __init__(self,x_len=None,**kwargs):
        super(FourierScaling, self).__init__(**kwargs)
        self.x_len = x_len
        
    
    def compute_energy(self,x,y_len_n,x_len_n,n_ch):
        '''Actually computes the sqrt of the energy
        '''
        x_comp = tf.complex(x[...,0],x[...,1])
        x_e = tf.square(tf.abs(x_comp))
        xy_n = y_len_n*x_len_n*np.sqrt(n_ch)
        if K.image_data_format() == 'channels_last': 
            if x_len_n % 2:  
                return tf.sqrt((tf.reduce_sum(x_e[...,0,:])
                        +2*tf.reduce_sum(x_e[...,1:,:]))/xy_n)
            else:
                return tf.sqrt((tf.reduce_sum(x_e[...,0,:])
                        +2*tf.reduce_sum(x_e[...,1:-1,:])
                        +tf.reduce_sum(x_e[...,-1,:]))/xy_n)
                
        elif K.image_data_format() == 'channels_first': 
            if x_len_n % 2:  
                return tf.sqrt((tf.reduce_sum(x_e[...,0])
                        +2*tf.reduce_sum(x_e[...,1:]))/xy_n)
            else:
                return tf.sqrt((tf.reduce_sum(x_e[...,0])
                        +2*tf.reduce_sum(x_e[...,1:-1])
                        +tf.reduce_sum(x_e[...,-1]))/xy_n)
    def call(self,x):
        x_shape = x.get_shape().as_list()
        assert x_shape[-1] == 2, "For FourierScaling, input should have last dim with len 2"
        if K.image_data_format() == 'channels_last': 
            nb,y_len,x_len,nch,_ = x_shape
        elif K.image_data_format() == 'channels_first': 
            nb,nch,y_len,x_len,_ = x_shape
            
        y_mesh,x_mesh =np.meshgrid(np.arange(y_len),np.arange(x_len))
        
        if self.x_len is None:
            x_len_n = ((x_len-1)*2)
        else:
            x_len_n = self.x_len
            
        y_mesh_f = y_mesh/(y_len)
        x_mesh_f = x_mesh/(x_len_n)
        
        freq_matrix = np.sqrt(np.square(y_mesh_f) + np.square(x_mesh_f))
        
        min_val = np.min(freq_matrix[freq_matrix>0])
        freq_matrix = np.transpose(np.maximum(freq_matrix,min_val))
        freq_matrix = freq_matrix[np.newaxis,:,:,np.newaxis,np.newaxis]
        
        prev_e = self.compute_energy(x,y_len,x_len_n,nch)
        norm_x = x * K.variable(1./(freq_matrix),dtype=K.floatx())
        
        new_e = self.compute_energy(norm_x,y_len,x_len_n,nch)
        norm_x = norm_x*prev_e/new_e
        
        norm_x.set_shape(x_shape)
        return norm_x
    def compute_output_shape(self,input_shape):
        return input_shape
    def get_config(self):
        config = {'x_len':self.x_len}
        base_config = super(FourierScaling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

###Not used since switched FourierScaling and irfft2d, kept for posterity.
#def tf_ifftshift(x,axes):
#    if isinstance(axes, integer_types):
#        axes = (axes,)
#    tmp = tf.convert_to_tensor(x)
#    y = tmp
#    for k in axes:
#        axes_shape = tf.shape(tmp)[k]
#        mid = axes_shape-(axes_shape+1)//2
#        mylist = tf.concat((tf.range(mid, axes_shape), tf.range(mid)),axis=0)
#        y = tf.gather(y, mylist,axis=k)
#    return y

class IRFFT2(Layer):
    def __init__(self,norm=False,**kwargs):
        super(IRFFT2, self).__init__(**kwargs)
        self.norm = norm
        
    def call(self,x):
        '''Expects 5-dim input: (batch,y,x,nch,complex components)
            TODO: Handle batch_size of more than 1 (currently, the norm factor will be off)
        '''
        x_shape = x.get_shape().as_list()
        assert x_shape[-1] == 2, "For IRFFT2, input should have last dim with len 2"
        real = x[...,0]
        imag = x[...,1]
        comp = tf.complex(real,imag)
        comp.set_shape(x_shape[:-1])
        
        if K.image_data_format() == 'channels_last':
            comp = tf.transpose(comp,[0,3,1,2])
            
        irfft = (tf.spectral.irfft2d(comp))
        i_shape= irfft.get_shape().as_list()
        
        if K.image_data_format() == 'channels_last':
            irfft = tf.transpose(irfft,[0,2,3,1])
            
        if self.norm:
            if K.image_data_format() == 'channels_last':
                norm = np.multiply(*i_shape[1:3])*np.sqrt(i_shape[-1])
            elif K.image_data_format() == 'channels_first':
                norm = np.multiply(*i_shape[-2:])*(np.sqrt(i_shape[1]))
            return irfft *np.sqrt(norm)
        else:
            return irfft
    
    def compute_output_shape(self,input_shape):
        if K.image_data_format() == 'channels_last': 
            nb,y_len,x_len,nch,_ = input_shape
            new_x_len = (x_len-1)*2
            return (nb,y_len,new_x_len,nch)
        elif K.image_data_format() == 'channels_first': 
            nb,nch,y_len,x_len,_ = input_shape
            new_x_len = (x_len-1)*2
            return (nb,nch,y_len,new_x_len)
    def get_config(self):
        return super(IRFFT2, self).get_config()
    
class RFFT2(Layer):
    def __init__(self,norm=False,**kwargs):
        super(RFFT2, self).__init__(**kwargs)
        self.norm = norm
    def call(self,x):
        '''Expects 4-dim input
            TODO: Handle batch_size of more than 1 (currently, the norm factor will be off)
        '''
        x_shape = x.get_shape().as_list()
        
        if K.image_data_format() == 'channels_last':
            x = tf.transpose(x,[0,3,1,2])
            
        rfft = (tf.spectral.rfft2d(x))
#        r_shape = rfft.get_shape().as_list()
        
        if K.image_data_format() == 'channels_last':
            rfft = tf.transpose(rfft,[0,2,3,1])
            
        real = tf.real(rfft)
        imag = tf.imag(rfft)
        new_x = tf.stack((real,imag),axis=-1)
        new_x.set_shape(self.compute_output_shape(x_shape))
        if self.norm:
            if K.image_data_format() == 'channels_last':
                norm = np.multiply(*x_shape[1:3])*np.sqrt(x_shape[-1])
            elif K.image_data_format() == 'channels_first':
                norm = np.multiply(*x_shape[2:])*np.sqrt(x_shape[1])
            return new_x / np.sqrt(norm)
        else:
            return new_x
    
    def compute_output_shape(self,input_shape):
        if K.image_data_format() == 'channels_last': 
            nb,y_len,x_len,nch = input_shape
            new_x_len = (x_len//2)+1
            return (nb,y_len,new_x_len,nch,2)
        elif K.image_data_format() == 'channels_first': 
            nb,nch,y_len,x_len = input_shape
            new_x_len = (x_len//2)+1
            return (nb,nch,y_len,new_x_len,2)

    def get_config(self):
        config = {'norm':self.norm}
        base_config =super(RFFT2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RandomRoll2D(Layer):
    def __init__(self,max_roll=None,
             **kwargs):
        super(RandomRoll2D, self).__init__(**kwargs)
        self.max_roll = max_roll
    def call(self,x):
        x_shape = x.get_shape()
        _,h,w,_ =  x_shape.as_list()
        if self.max_roll is None:
            y_roll = tf.random_uniform([],0,
                                maxval=h,dtype=tf.int32)
            x_roll = tf.random_uniform([],0,
                                        maxval=w,dtype=tf.int32)
        else:
            y_roll = tf.random_uniform([],0,maxval=self.max_roll,
                                       dtype=tf.int32)
            x_roll = tf.random_uniform([],0,maxval=self.max_roll,
                                        dtype=tf.int32)
        if K.image_dim_ordering() =='tf':
            x= tf.concat([x[:,y_roll:,:,:], x[:,:y_roll,:,:]], axis=1)
            x= tf.concat([x[:,:,x_roll:,:], x[:,:,:x_roll,:]], axis=2)
        else:
            x= tf.concat([x[:,:,y_roll:,:], x[:,:,:y_roll,:]], axis=2)
            x= tf.concat([x[:,:,:,x_roll:], x[:,:,:,:x_roll]], axis=3)
        x.set_shape(x_shape)
        return x
    def get_config(self):
        config = {'max_roll':self.max_roll}
        base_config =super(RandomRoll2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

class Transpose(Layer):
    def __init__(self,perm,conjugate=False,
         **kwargs):
        super(Transpose, self).__init__(**kwargs)
        self.perm = perm
        self.conjugate = conjugate
    def call(self,x):
        return tf.transpose(x,perm=self.perm,conjugate=self.conjugate)
    def compute_output_shape(self,input_shape):
        new_shape = []
        for idx in self.perm:
            new_shape.append(input_shape[idx])
        return tuple(new_shape)
    def get_config(self):
        config = {'perm':self.perm,'conjugate':self.conjugate}
        base_config =super(Transpose, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

'''See Resize2D
class BilinearResize(Layer):
        
    def __init__(self,target_energy,
         **kwargs):
        super(BilinearResize, self).__init__(**kwargs)
    def call(self,x):
        pass
    def compute_output_shape(self,input_shape):
        pass
    def get_config(self):
        config = {}
        base_config =super(BilinearResize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
'''
class NNResize(Layer):
        
    def __init__(self,new_size,
         **kwargs):
        super(NNResize, self).__init__(**kwargs)
        assert K.image_data_format() == 'channels_last', 'requires channels last ordering'
        self.new_size = new_size
    def call(self,x):
        return tf.image.resize_nearest_neighbor(x,self.new_size)
    def compute_output_shape(self,input_shape):
        new_shape = (input_shape[0],
                     self.new_size[0],
                     self.new_size[1],
                     input_shape[3])
        return new_shape
    def get_config(self):
        config = {'new_size':self.new_size}
        base_config =super(NNResize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class Gather(Layer):
    def __init__(self,indices,axis,**kwargs):
        super(Gather, self).__init__(**kwargs)
        self.axis = axis
        self.indices=indices
    def call(self,x):               
        x = tf.gather(x,axis=self.axis,indices=self.indices)
        return x
    def compute_output_shape(self, input_shape):
        new_shape = list(input_shape)
        new_shape[self.axis] = len(self.indices)
        return tuple(new_shape)
    def get_config(self):
        config = {'axis':self.axis,'indices':self.indices}
        base_config =super(Gather, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
tf_layers_dict = globals()