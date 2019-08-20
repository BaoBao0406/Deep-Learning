# CNN Model 2 - two layers
# Load Keras and other function
from keras.models import Sequential, Model
from keras.layers import Conv2D
from keras.utils import np_utils

# Load and display SVG function
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import os
os.environ["PATH"] += os.pathsep + 'C:/Users/Patrick/.conda/pkgs/graphviz-2.38.0-0/Library/bin/'

# Create two convolutional neural network model
model = Sequential()
model.add(Conv2D(filters=3, kernel_size=(3,3), input_shape=(6,6,1), name='Conv2D_1'))
model.add(Conv2D(filters=2, kernel_size=(3,3), name='Conv2D_2'))

# Use SVG format to display
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
