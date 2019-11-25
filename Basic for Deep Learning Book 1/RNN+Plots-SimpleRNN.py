# RNN Model 1 - SimpleRNN
from keras.models import Model
from keras.layers import Input, SimpleRNN

# Load SVG and other functions
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import os
os.environ["PATH"] += os.pathsep + 'C:/Users/Patrick/.conda/pkgs/graphviz-2.38.0-0/Library/bin/'

# Set unit, step, dimension and data type
units = 10
time_steps = 5
input_dim = 15
input_shape = (time_steps, input_dim)

# Create Recurrent Neural Network
x = Input(shape=input_shape, name='Input')
y = SimpleRNN(units=units, activation='sigmoid', name='SimpleRNN_1')(x)
model = Model(input=[x], outputs=[y])

# Use SVG format
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

# Create Recurrent Neural Network Model and output data
y = SimpleRNN(units=units, activation='sigmoid', return_sequences=True, name='SimpleRNN_1')(x)
model = Model(inputs=[x], outputs=[y])

# Use SVG to display model
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

# Create Recurrent Neural Network Model and output status
y, state = SimpleRNN(units=units, activation='sigmoid', return_state=True, name='SimpleRNN_1')(x)
model = Model(inputs=[x], outputs=[y])

# Use SVG to display model
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

