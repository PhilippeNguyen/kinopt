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
            self.jitter = jitter

    
    def call(self,x):
        _,h,w,_ =  x.get_shape().as_list()
        x = tf.image.resize_image_with_crop_or_pad(x,h+self.jitter[0],
                             w+self.jitter[1])
        y_shift_1 = tf.random_uniform([],minval=0,
                                    maxval=(self.jitter[0]//2),
                                    dtype=tf.int32)
        x_shift_1 = tf.random_uniform([],minval=0,
                                    maxval=(self.jitter[1]//2),
                                    dtype=tf.int32)
        x = tf.image.crop_to_bounding_box(x,y_shift_1,x_shift_1,h,w)
        return x
    
    def get_config(self):
        config = {'jitter': self.jitter}
        base_config = super(Jitter2D, self).get_config()
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
             **kwargs):
        super(RandomRotate2D, self).__init__(**kwargs)
        if rotate_vals is None:
                rotate_vals = [-5.,-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.,5.]
        self.rotate_vals = rotate_vals
        self.logs = [1. for _ in self.rotate_vals]
        self.rotate_vals_tf =  tf.convert_to_tensor(self.rotate_vals)

    def call(self,x):
        rotate_sample = tf.multinomial(tf.log([self.logs]), 1)
        rotate = self.rotate_vals_tf[tf.cast(rotate_sample[0][0], tf.int32)]
        x = tf.contrib.image.rotate(x,rotate)
        return x

    def get_config(self):
        config = {'rotate_vals': self.rotate_vals}
        base_config = super(RandomRotate2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
#No Longer Tested
#class Cholesky2D(Layer):
#    def __init__(self,axis=3,**kwargs):
#        super(Cholesky2D, self).__init__(**kwargs)
#        self.axis=axis
#        
#    def call(self,x):
#        x_shape = x.get_shape()
#        ndim= len(x_shape.as_list())
#        cov = self.channel_cov(x)
#        chol = tf.linalg.cholesky(cov)
#        inv_chol = tf.linalg.inv(chol)
#        x = self.rollaxes(x,self.axis,0,ndim)
#        x = tf.tensordot(inv_chol,x,axes=1)
#        x = self.rollaxes(x,0,self.axis,ndim)
#
#        x.set_shape(x_shape)
#        return x
#    
#    
#    def channel_cov(self,x):
#        '''computes covariance along the given axis
#        '''
#        x_shape = x.get_shape().as_list()
#        ndim= len(x_shape)
#        x = self.rollaxes(x,self.axis,ndim-1,ndim)
#        x_flat = tf.reshape(x,[-1,x_shape[-1]])
#        x_mean = tf.expand_dims(tf.reduce_mean(x_flat,axis=0),axis=0)
#        X = x_flat - x_mean
#        len_x = X.get_shape().as_list()[0]
#        X_t = tf.transpose(X)
#        return tf.tensordot(tf.conj(X_t),X,axes=1)/(len_x-1)
#    
#    def rollaxes(self,x,start_pos,end_pos,ndim=None):
#        if not ndim:
#            ndim= len(x.get_shape().as_list())
#        roll_indices = list(range(ndim))
#        tmp = roll_indices.pop(start_pos)
#        roll_indices.insert(end_pos,tmp)
#        return tf.transpose(x,roll_indices)
#        
#    def get_config(self):
#        config = {'axis':self.axis}
#        base_config = super(Cholesky2D, self).get_config()
#        return dict(list(base_config.items()) + list(config.items()))

class ChannelDecorrelate(Layer):
    def __init__(self,whitening_matrix=None,**kwargs):
        super(ChannelDecorrelate, self).__init__(**kwargs)
        if whitening_matrix is None:
            whitening_matrix = np.array(
                                [[ 0.64636492,  0.,          0.        ],
                                 [-0.74520612,  0.67786914,  0.        ],
                                 [ 0.16395189, -0.5827935,   0.2863121 ]],
                                dtype=np.float32)
                                        
            self.whitening_matrix = whitening_matrix
        elif type(whitening_matrix) is np.ndarray:
            self.whitening_matrix = np.float32(whitening_matrix)
        else:
            #TODO: Check if tensorflow tensor
            raise Exception('whitening_matrix must be a numpy array')

        
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
    
tf_layers_dict = globals()