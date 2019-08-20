# CNN model 7 - Flatten and Dense
# Load Keras and other function
from keras.models import Sequential, Model
from keras.layers import Dense, Acitivation, Flatten, Conv2D
from keras.utils import np_utils

# Load SVG functions
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import os
os.environ["PATH"] += os.pathsep + 'C:/Users/Patrick/.conda/pkgs/graphviz-2.38.0-0/Library/bin/'

# Create Neural Network for output to 1 dimension for Fully Connected Layer
model = Sequential()
model.add(Conv2D(filters=3, kernel_size=(3, 3), input_shape=(6, 6, 1), padding='same', name='Conv2D_1'))
model.add(Flatten(name='Flatten_1'))
model.add(Dense(units=10, activation='softmax', name='Dense_1'))

# Use SVG format to display
SVG(model_to_dot(model, show_shape=True).create(prog='dot', format='svg'))
