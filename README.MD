# Kinopt
Input optimization with keras models. This can be used for feature visualization, neural style transfer,adversarial examples, and other stuff.

|  |  |  |
| --- | --- | --- |
| ![featvis 1](https://github.com/PhilippeNguyen/kinopt/blob/master/assets/feat_vis_0.png "featvis 1") | ![featvis 2](https://github.com/PhilippeNguyen/kinopt/blob/master/assets/feat_vis_40.png "featvis 2") | ![featvis 3](https://github.com/PhilippeNguyen/kinopt/blob/master/assets/feat_vis_170.png "featvis 3") |

This project provides 3 main things:
*  A way of inserting layers into an already trained keras models**
*  New layers which facilitate feature visualization and other types on input optimization.
*  A way of using keras optimizers to optimize your input tensors**

** Note: Most of this project was done awhile back and used kind of hack-y ways to accomplish some
of these goals. Since then, I've figured out how to do some of these with mostly standard
Keras/Tensorflow machinery. Both the old and the new ways are explained in featvis_long_tutorial.py

With this, you will be able to compare the feature visualization of any keras model see [this article](https://distill.pub/2017/feature-visualization/) for more info. For example, you can have different models draw bees:

| VGG19 | GoogleNet | ResNet50 |
| --- | --- | --- |
|![VGG19 Bees](https://github.com/PhilippeNguyen/kinopt/blob/master/assets/bee_vgg19.png "VGG19 Bees") | ![GoogleNet Bees](https://github.com/PhilippeNguyen/kinopt/blob/master/assets/bee_googlenet.png "GoogleNet Bees")| ![ResNet50 Bees](https://github.com/PhilippeNguyen/kinopt/blob/master/assets/bee_resnet50.png "ResNet50 Bees") |

## Requirements
The code works for model trained in keras (any backend), however to run this code you'll need to use the tensorflow backend.

*  tensorflow
*  keras
*  imageio
*  scikit-imageio

## Installation
Standard installation, git clone this repo
```
git clone https://github.com/PhilippeNguyen/kinopt.git
```
then pip install
```
pip install -e ./kinopt
```

## Usage

See featvis_example.py for a quick script showing how to do feature visualization.

|  |  |  |
| --- | --- | --- |
| ![featvis extra 1](https://github.com/PhilippeNguyen/kinopt/blob/master/assets/feat_vis_extra_1.png "featvis extra 1") | ![featvis extra 2](https://github.com/PhilippeNguyen/kinopt/blob/master/assets/feat_vis_extra_2.png "featvis extra 2") | ![featvis extra 3](https://github.com/PhilippeNguyen/kinopt/blob/master/assets/feat_vis_extra_3.png "featvis extra 3") |

Explanation of some of the script arguments:
*  --model: path to a saved keras model. Models are saved with the model.save function. Doesn't work if you only save the weights using the Model.save_weights function. Note that you'll want to have the pointwise activations (relu/softmax/...) separated from the linear transforms (Dense/Convs/...), my script at /other_scripts/separate_activations.py should do this for you. Keras gives you access to a bunch of classification models, see [keras_applications](https://github.com/keras-team/keras-applications) (comes installed with keras). Every one of these models should work, though I couldn't try NASNetLarge since it was too big to download.
*  --preprocess_mode: the preprocessing function to use. See [keras_applications.imagenet_utils.preprocess_input](https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py) for the different types of modes available. If you're using a keras_application model, then they'll be using one of the modes, and you'll probably need to dig around the source code on github to see what it is.
*  --layer_identifier: Either the layer name or the layer index in Model.get_config['layers']
*  --neuron_index: This is the channel index of the tensor corresponding to the layer identified above
*  --output: name/path of the output png to save the optimized input image.

See featvis_long_tutorial.py for long explanation of what's going on, as well as alternate methods of doing input optimization in keras so you won't have to deal with my janky code.
Note that optimizing in the fourier space (as mentioned [here](https://distill.pub/2017/feature-visualization/)) is implemented, but not shown in the feat vis exmaples. I also add a random roll transformation which is not mentioned in the article.

You can also do neural style transfer, see neural_style_example.py

![Neural Style Example](https://github.com/PhilippeNguyen/kinopt/blob/master/assets/neural_style_1.png "Neural Style Example")
