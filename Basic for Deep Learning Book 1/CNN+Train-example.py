# CNN Train 1
# Load Keras and other functions
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activagtion, Flatten, Conv2D
from keras.utils import np_utils
from keras.optimizers import SGD

# Load  Callbacks and data from Numpy
import keras.callbacks as callbacks
import numpy as np

# Display SVG functions
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import os
os.environ["PATH"] += os.pathsep + 'C:/Users/Patrick/.conda/pkgs/graphviz-2.38.0-0/Library/bin/'

# Prepare random number for Train data
x_train = np.random.random((100, 6, 6, 1))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20, 6, 6, 1))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

# Create Neural Network model
model = Sequential()
model.add(Conv2D(filters=3, kernel_size=(3, 3), input_shape=(6, 6, 1), kernel_initializer='lecun_uniform', name='Conv2D_1'))
model.add(Flatten(name='Flatten_1'))
model.add(Dense(units=10, activation='softmax', name='Dense_1'))

# Output Sequential
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

# Set to End
earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Edit Model
model.compile(loss='mean_squared_error', optimizer='sgd')
model.fit(x_train, y_train, batch_size=32, epochs=10, callbacks=[earlyStopping], validation_split=0.2)
