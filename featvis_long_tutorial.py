import imageio
import keras
import keras.backend as K
import kinopt
import argparse
import numpy as np
import os
from kinopt.layers.tf_layers import (tf_layers_dict,
                                     ChannelDecorrelate,RandomRoll2D,
                                     Jitter2D,RandomResize2D,RandomRotate2D)
from kinopt.layers import LogisticTransform,ImagenetPreprocessorTransform
fs = os.path.sep
K.set_learning_phase(False)



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', action='store', dest='model',
                    required=True,
        help='path to a keras hdf5 model')
parser.add_argument('--output', action='store', dest='output',
                    required=True,
        help='output path,name of the output png')
parser.add_argument('--layer_identifier', action='store',
                    dest='layer_identifier',
                    required=True,
                    help='layer to activate, either name or index')
parser.add_argument('--neuron_index', action='store',
                    dest='neuron_index',
                    default=0,type=int,
                    help=('Activity of the neuron index (channel) in the given layer'
                          ' to optimize'))

parser.add_argument('--image_size', action='store',
                    dest='image_size',
                    default=224,type=int,
                    help='height and width of the image')
parser.add_argument('--num_iter', action='store',
                    dest='num_iter',
                    default=1000,type=int,
                    help='number of optimization iterations')
parser.add_argument('--preprocess_mode', action='store',
                    dest='preprocess_mode',required=True,
                    help='this is the preprocess mode the same as '
                    'keras-applications.imagenet_utils.preprocess_input.'
                    ' Which usually defaults to "caffe" ')
parser.add_argument('--load_method', action='store',
                    dest='load_method',default='layer_list',
                    help='What load method to use, (either "layer_list", '
                    '"config_dict" or "tensor"'
                    '')
parser.add_argument('--fit_method', action='store',
                    dest='load_method',default='layer_list',
                    help='What load method to use, (either "layer_list", '
                    '"config_dict" or "tensor"'
                    '')
'''Note: it is recommended to maximize the response of a neuron _before_ activation,
    If your model has it's activations not seperated from the linear layer, use the
    separate_activations.py script found in other_scripts.
'''

args = parser.parse_args()
output = args.output if args.output.endswith('.png') else args.output+'.png'

try:
    layer_identifier = int(args.layer_identifier)
except:
    layer_identifier = args.layer_identifier

#Initialize input, randomly
image_size =args.image_size
img_shape = (1,image_size,image_size,3)
init_img = 0.1*np.random.randn(*img_shape)

"""Load the model, where kinopt starts being used.
   We can load the model, while also inserting keras layers.
   3 loading methods, all of them essentially connect tensors to the input 
   of the model.   
   
   1. "tensor", just use keras' standard mechanism for building a model on 
   top of a specified tensor. This is actually the cleanest way of adding input
   operations, just uses the keras methods. I wrote 2. and 3. because I didn't know
   about 1. This method will work as long as it works in keras. 
   One caveat about method 1. is that you no longer have one unified keras.Model
   
   2. "layer_list", 
   3. "config_dict",   
   Both 2. and 3. require a list of lists. That is, a list for each input to the model.
   Since we only have one input to our model, we have an outer list of len 1.
   The inner list represents a sequence of layers that are applied to the input.
   The code assumes the output of one is fed to the input of another, so merge 
   layers will probably not work.
   In 2.,"layer_list",  you instantiate the layers and have them as part of the list.
   In 3.,"config_dict", the layers are defined by config dicts, (the same as you get
   when you do a model.get_config()))
   
   
"""
if args.load_method == "tensor":
    input_layer = keras.layers.Input(batch_shape=img_shape)
    x = ChannelDecorrelate()(input_layer)
    x = LogisticTransform(scale=255,name='png_layer')(x)
    x = ImagenetPreprocessorTransform(mode=args.preprocess_mode)(x)
    x = RandomRoll2D()(x)
    x = Jitter2D(jitter=16)(x)
    x = RandomResize2D(resize_vals=np.arange(0.75,1.25,0.025))(x)
    x = RandomRotate2D(rotate_vals=np.arange(-45.,45.))(x)
    x = Jitter2D(jitter=8)(x)
    
    preproc_model = keras.models.Model(input_layer,x)
    loaded_model = keras.models.load_model(args.model,compile=False,
                                           custom_objects=tf_layers_dict)
    fit_tensor = kinopt.utils.get_layer_output(loaded_model,layer_identifier)
    fit_tensor_model = keras.models.Model([loaded_model.input],[fit_tensor])
    
    fit_tensor = fit_tensor_model(preproc_model.output)
    input_tensor = input_layer
    png_layer = preproc_model.get_layer('png_layer').output
    

elif args.load_method == "layer_list":
    added_layers = [[ChannelDecorrelate(),
                      LogisticTransform(scale=255,name='png_layer'),
                      ImagenetPreprocessorTransform(mode=args.preprocess_mode),
                      RandomRoll2D(),
                      Jitter2D(jitter=16),
                      RandomResize2D(resize_vals=np.arange(0.75,1.25,0.025)),
                      RandomRotate2D(rotate_vals=np.arange(-45.,45.)),
                      Jitter2D(jitter=8)
                    ]]
    model = kinopt.models.load_model(args.model,initial_inputs=init_img,
                                     inserted_layers=added_layers,
                                     custom_objects=tf_layers_dict)
    
    input_tensor = model.input
    png_layer = model.get_layer('png_layer').output
    fit_tensor = kinopt.utils.get_layer_output(model,layer_identifier)
    
elif args.load_method == "config_dict":
    added_layers = [[
       {
        "class_name":"ChannelDecorrelate",
        },
       {
        "class_name":"LogisticTransform",
        "config":{'scale':255.,
                  'name':'png_layer'},
          'name':'png_layer'
        },
       {
        "class_name":"ImagenetPreprocessorTransform",
        "config":{'mode':args.preprocess_mode}
        },
        {
        "class_name":"RandomRoll2D",
        },
        {
        "class_name":"Jitter2D",
        "config":
          {
            "jitter":16,
          },
        },
        {
        "class_name":"RandomResize2D",
        "config":
          {
            "resize_vals":np.arange(0.75,1.25,0.025),
          },
        },
        {
        "class_name":"RandomRotate2D",
        "config":
          {
            "rotate_vals":np.arange(-45.,45.),
          },
        },
        {
        "class_name":"Jitter2D",
        "config":
          {
            "jitter":8,
          },
        }
      ]
    ]
    model = kinopt.models.load_model(args.model,initial_inputs=init_img,
                                 inserted_layers=added_layers,
                                 custom_objects=tf_layers_dict)
    
    input_tensor = model.input
    png_layer = model.get_layer('png_layer').output
    fit_tensor = kinopt.utils.get_layer_output(model,layer_identifier)
else:
    raise Exception("load_method not understood")

#Function for getting the output from the 'png_layer', will be used after fitting
png_func = K.function([input_tensor],
                      [png_layer])

#compile (make loss, make optimizer)
neuron = kinopt.utils.get_neuron(fit_tensor,
                                     feature_idx=args.neuron_index)
loss = -(K.mean(neuron))
optimizer = keras.optimizers.Adam(lr=0.05)


'''Here we actually optimize the input image, this can actually be done in two ways.

    If the input to optimize (input_tensor) is a placeholder, use kinopt.fitting.input_fit
    If the input to optimize is a variable, use kinopt.fitting.input_fit_var, this 
        can actually be done without kinopt.
'''
out = kinopt.fitting.input_fit(input_tensor,loss,optimizer,
                               init_img,num_iter=args.num_iter)
proc_img = png_func([out])[0]
imageio.imsave(output,proc_img[0])