#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 18:18:23 2017

Tensorflow does not allow updates to tf.Tensors in the same way that keras
usually does it for variables
These optimizers are based off of the Keras optimizers but separate out the grad
update from the optimizer update.


"""

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
        if (update.__class__.__name__ =='Tensor'
            and update.op.inputs[0].name == model_input.name):
            updated_param = updates.pop(idx)
            break
    grad = updated_param - model_input
    return grad,updates

