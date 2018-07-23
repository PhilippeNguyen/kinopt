#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 18:18:23 2017

Tensorflow does not allow updates to tf.Tensors in the same way that keras
usually does it. 
These optimizers are based off of the Keras optimizers but separate out the grad
update from the optimizer update.


"""
from keras.optimizers import Optimizer
import keras.backend as K

def get_input_updates(optimizer,loss,model_input):
    '''Extremely hack-y method of pulling out the p (params) value from the 
        optimizer. This needs to be done because tf cannot update Placeholders/Tensors
        normally.
        We will return the updates with the params removed, as well as the tensor
        corresponding to the param update, the param update will be done outside
        of the usual graph computation.
    '''
    def fake_assign(new_val,name=None):
        return new_val
    setattr(model_input,'assign',fake_assign)
    updates = optimizer.get_updates(loss=[loss],
                                     params=[model_input],
                                     )
    for idx,update in enumerate(updates):
        if update.op.inputs[0].name == model_input.name:
            updated_param = updates.pop(idx)
            break
    grad = updated_param - model_input
    return grad,updates

class FGS(Optimizer):
    '''Fast Gradient Sign
        Doesn't really optimize, but uses the same design as optimizers.
        Make sure to use num_iter = 1 for when using input_fit
        
    '''
    def __init__(self,eps,**kwargs):
        super(FGS, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
        self.eps = eps
    def get_updates(self,loss,params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        self.weights = [self.iterations]
        
        for p, g in zip(params, grads):
            new_p = p +self.eps*K.sign(g)
            self.updates.append(K.update(p, new_p))
        return self.updates
    def get_config(self):
        config = {'eps':self.eps }
        base_config = super(FGS, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))