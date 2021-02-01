import numpy as np

# sigmoid function
def sigmoid(x):
    # implemented in a way where if x is too big, we won't have runtime warning
    if x >= 0:
        return 1 / (1 + np.exp(-x))       
    return np.exp(x)/ (1 + np.exp(x))


# softmax function
def softmax(x):
    # implemented in a way where if x is too big, we won't have runtime warning
    x -= np.max(x)
    exp = np.exp(x)
    return exp/ np.sum(exp, axis = 1, keepdims = True)


# ReLU function, use He initialization
def relu(x):
    if x > 0:
        return x
    return 0


# Leaky ReLU function
def leakyRelu(x):
    if x > 0:
        return x
    return 0.1 * x  #change coefficient for leakiness level


# tanch function, use Xavier initialization
def tanh(x):
    # implemented in a way where if x is too big, we won't have runtime warning
    # np.tanh(x)
    if x >= 0:
        return (1 - np.exp(-2*x))/ (1 + np.exp(-2*x))       
    return (np.exp(2 * x) - 1)/ (np.exp(2 * x) + 1)

