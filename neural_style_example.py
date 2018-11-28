import argparse
import kinopt
import imageio
from skimage.transform import resize
import numpy as np
import keras
from kinopt.utils.tensor_utils import compare_external_input
from kinopt.layers.base_layers import (ImagenetPreprocessorTransform,
                                       LogisticTransform)
import keras.backend as K
from scipy.special import logit
import tensorflow as tf

def logit_preprocess(x):
    x = np.clip(x,1,254)
    return logit(x/255.)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', action='store', dest='model',
                    required=True,
        help='model path')
parser.add_argument('--style_layers', action='store',
                    dest='style_layers',nargs='+',
                    required=True,
                    help='layers to use for style loss, either name or index')
parser.add_argument('--content_layers', action='store',
                    dest='content_layers',nargs='+',
                    required=True,
                    help='layers to use for content loss, either name or index')
parser.add_argument('--content_image', action='store',
                    dest='content_image',required=True,
                    help='path to the content image')
parser.add_argument('--style_image', action='store',
                    dest='style_image',required=True,
                    help='path to the style image')
parser.add_argument('--preprocess_mode', action='store',
                    dest='preprocess_mode',required=True,
                    help='this is the preprocess mode the same as '
                    'keras-applications.imagenet_utils.preprocess_input.'
                    ' Which usually defaults to "caffe" ')
parser.add_argument('--content_weight', type=float, 
                    default=0.025, 
                    help='Content weight.')
parser.add_argument('--style_weight', type=float,
                    default=0.1, 
                    help='Style weight.')
parser.add_argument('--tv_weight', type=float, 
                    default=1.0, 
                    help='Total Variation weight.')
parser.add_argument('--num_iter', action='store',
                    dest='num_iter',
                    default=500,type=int,
                    help='number of optimization iterations')
parser.add_argument('--output', action='store',
                    dest='output',
                    required=True,
                    help='output path,name of the output png')
parser.add_argument('--new_output_layers', action='store',
                    dest='new_output_layers',nargs='+',
                    default=None,
                    help='remove any layers that come after these layers')

parser.add_argument('--new_size', action='store',
                    dest='new_size',
                    default=None,type=int,
                    help='resize the longest edge of the image to this size')

args = parser.parse_args()
output = args.output if args.output.endswith('.png') else args.output + '.png'
new_size = args.new_size

#need to preprocess the content and style images
content_img = np.float32(imageio.imread(args.content_image))
if new_size is not None:
    scale = new_size/max(content_img.shape[:2])
    new_shape = [int(shape*scale) for shape in content_img.shape[:2]]
    content_img = resize(content_img,new_shape,preserve_range=True)
content_img = logit_preprocess(content_img)

style_img = np.float32(imageio.imread(args.style_image))
style_img = resize(style_img,content_img.shape[:2],preserve_range=True)
style_img = logit_preprocess(style_img)

content_img = np.expand_dims(content_img,axis=0)
style_img = np.expand_dims(style_img,axis=0)

init_img = content_img.copy()
    
#set up and load the model
added_layers = [[
                LogisticTransform(scale=255,name='png_layer'),
                ImagenetPreprocessorTransform(mode=args.preprocess_mode),
                ]]
model = kinopt.models.load_model(args.model,initial_inputs=init_img,
                                 inserted_layers=added_layers,
                                 new_output_layers=args.new_output_layers)

png_layer = model.get_layer('png_layer').output

png_func = K.function([model.input],
                      [png_layer])

#generate the loss
loss = 0
num_content = float(len(args.content_layers))
for content_layer in args.content_layers:
    fit_tensor,compare_tensor = compare_external_input(model,compare_input=content_img,
                                             layer_identifier=content_layer)
    content_loss = kinopt.losses.tensor_sse(fit_tensor,compare_tensor)
    loss += (args.content_weight/num_content)*content_loss

num_style = float(len(args.style_layers))
for style_layer in args.style_layers:
    fit_tensor,compare_tensor = compare_external_input(model,compare_input=style_img,
                                           layer_identifier=style_layer)
    style_loss = kinopt.losses.style_loss(fit_tensor,compare_tensor,
                                          norm_channels=False)
    loss += (args.style_weight/num_style)*style_loss
    
tv_loss = kinopt.losses.spatial_variation(model.input,power=1.25)
loss += args.tv_weight*tv_loss

#Set the optimizer and fit
optimizer = keras.optimizers.Adam(lr=0.05)
out = kinopt.fitting.input_fit(model.input,loss,optimizer,
                               init_img,num_iter=args.num_iter)
proc_img = png_func([out])[0]
imageio.imsave(output,proc_img[0])

