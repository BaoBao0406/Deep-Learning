# CNN Model 4 - two strides
# Load Keras and other functions
from keras.models import Sequential, Model
from keras.layers import Conv2D
from keras.utils import np_utils

# Load SVG functions
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import os
os.environ["PATH"] += os.pathsep + 'C:/Users/Patrick/.conda/pkgs/graphviz-2.38.0-0/Library/bin/'

# Create stride=2 for Convolutional Neural Network model
model = Sequential()
model.add(Conv2D(filters=3, kernel_size=(3,3), input_shape=(6,6,1), strides=2, name='Conv2d_1'))

# Use SVG format to display model
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
