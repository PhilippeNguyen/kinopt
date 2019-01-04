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
from kinopt.weightnorm import AdamWithWeightnorm
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
parser.add_argument('--preprocess_mode', action='store',
                    dest='preprocess_mode',required=True,
                    help='this is the preprocess mode the same as '
                    'keras-applications.imagenet_utils.preprocess_input.'
                    ' Which usually defaults to "caffe" ')
parser.add_argument('--image_size', action='store',
                    dest='image_size',
                    default=224,type=int,
                    help='height and width of the image')
parser.add_argument('--num_iter', action='store',
                    dest='num_iter',
                    default=1000,type=int,
                    help='number of optimization iterations')


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


#set up the model, add extra layers
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
    

#Function for getting the output from the 'png_layer', will be used after fitting
png_func = K.function([input_tensor],
                      [png_layer])

#compile (make loss, make optimizer)
neuron = kinopt.utils.get_neuron(fit_tensor,
                                     feature_idx=args.neuron_index)
loss = -(K.mean(neuron))
optimizer = keras.optimizers.Adam(lr=0.05)

#you can also try using a weight normalized optimizer
#optimizer = AdamWithWeightnorm(lr=0.05)

#fit
out = kinopt.fitting.input_fit(input_tensor,loss,optimizer,
                               init_img,num_iter=args.num_iter)
proc_img = png_func([out])[0]
imageio.imsave(output,proc_img[0])
