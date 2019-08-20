# CNN Model 9 - Multi Sigmoid
# Load Keras and other functions
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.utils import np_utils

# Display SVG and functions
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import os
os.environ["PATH"] += os.pathsep + 'C:/Users/Patrick/.conda/pkgs/graphviz-2.38.0-0/Library/bin/'

# Create Neural Network > Flatten > Full Layer for Multi model
model = Sequential()
model.add(Conv2D(filters=3, kernel_size=(3, 3), input_shape=(6, 6, 1), padding='same', name='Conv2D_1'))
model.add(Flatten(name='Flatten_1'))
model.add(Dense(units=10, activagtion='sigmoid', name='Dense_1'))
