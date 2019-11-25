# RNN Model 2 - LSTM
from keras.models import Model
from keras.layers import Input, LSTM

# Load SVG and other functions
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import os
os.environ["PATH"] += os.pathsep + 'C:/Users/Patrick/.conda/pkgs/graphviz-2.38.0-0/Library/bin/'

# Set unit, step, vector and data shape
units = 10
time_steps = 5
input_dim = 15
input_shape = (time_steps, input_dim)

# Create Recurrent Neural Network
x = Input(shape=input_shape, name='Input')
y = LSTM(units=units, activation='sigmoid', name='LSTM_1')(x)
model = Model(inputs=[x], outputs=[y])

# Use SVG to display
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

# Use Recurrent Neural Network model to ouput 
y = LSTM(units=units, activation='sigmoid', return_sequences=True, name='LSTM_1')(x)
model = Model(inputs=[x], outputs=[y])

# Use SVG to display
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

# Create Recurrent Neural Network model and output status
x = Input(shape=input_shape, name='Input')
y, state_1, state_2 = LSTM(units=units, activation='sigmoid', return_state=True, name='LSTM_1')(x)
model = Model(inputs=[x], outputs=[y])

# Use SVG format to display
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

