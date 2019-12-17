import numpy as np
import matplotlib.pylab as plt

# 1. Step_function
# Create step function
def step_function1(x):
    if x > 0:
        return 1
    else:
        return 0

# Re-create step function to show integer
def step_function2(x):
    y = x > 0
    return y.astype(np.int)

# Display the diagram for Sigmoid
def step_function(x):
    return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()


# 2. Sigmoid function
# Create Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()


# 3. ReLU function
def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-0.5, 5)
plt.show()