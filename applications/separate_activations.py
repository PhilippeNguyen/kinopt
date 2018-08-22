import keras
import argparse
import tempfile
import copy
import os


'''Script for separating activations from layers, saves a new model
'''



def get_activation_names(layers):
    act_names = set()
    for layer in layers:
        act_names.add(layer['name'])
    return act_names
def create_activation_layer(activation,name,input_name):
    config = {'name':name,
              'activation':activation,
              'trainable':True}
    layer = {'class_name':'Activation',
             'config':config,
             'inbound_nodes':[[[input_name,0,0,{}]]],
             'name':name}
    return layer
    
def sep_config(model_config):
    model_config = copy.deepcopy(model_config)
    layers = model_config['layers']
    activation_names = get_activation_names(layers)
    
    act_id = 0
    new_layers = []
    redirect_dict = {}
    new_act_names = set()
    for idx,layer in enumerate(layers):
        config = layer['config']
        if 'activation' in config and config['activation'] != 'linear':
            #add only layer with activation removed from 
            mod_layer = copy.deepcopy(layer)
            mod_layer['config']['activation'] = 'linear'
            new_layers.append(mod_layer)
            
            #get new activation name
            while True:
                act_name = 'activation_'+str(act_id)
                if act_name in activation_names:
                    act_id+=1
                else:
                    activation_names.add(act_name)
                    break
            #add activation config 
            act_layer = create_activation_layer(config['activation'],
                                                act_name,
                                                layer['name'])
            new_layers.append(act_layer)
            #remember the name of new activations, will swap names in inbound_nodes
            redirect_dict[layer['name']] = act_name
            new_act_names.add(act_name)
        else:
            new_layers.append(layer)
            
    #swap out inbound_nodes with new activation names
    for layer in new_layers:
        if layer['name'] in new_act_names:
            continue
        inbound_nodes = layer['inbound_nodes']
        for outer_node in inbound_nodes:
            for inner_node in outer_node:
                name = inner_node[0]
                if name in redirect_dict:
                    inner_node[0] = redirect_dict[name]
                
    #swap out output_names with new activation names
    for inner_node in model_config['output_layers']:
        name = inner_node[0]
        if name in redirect_dict:
            inner_node[0] = redirect_dict[name]
            
    model_config['layers'] = new_layers
    return model_config
        

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', action='store', dest='model',
                    required=True,
                    help='Path to the model hdf5 file whose activations will be separated into separate layers')
parser.add_argument('--output', action='store', dest='output',
                    required=True,
                    help='name of output hdf5, to be saved')
args = parser.parse_args()
output = args.output if args.output.endswith('.hdf5') else args.output + '.hdf5'

model = keras.models.load_model(args.model)
_,tmp_path = tempfile.mkstemp(suffix='.hdf5')
model.save_weights(tmp_path)
config = model.get_config()
new_config = sep_config(config)
model_config = {'class_name':'Model','config':new_config}
new_model = keras.models.model_from_config(model_config)
new_model.load_weights(tmp_path)
new_model.save(output)

os.remove(tmp_path)