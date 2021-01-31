import numpy as np
from activations import *



# forward propogate, predict using the current weights
def forward_propogation(X, weights, activations):
    layers = len(weights) // 2      # number of layers (excluding input layer)
    A = X       # A will be activation function

    activ_vals = {"L0": A}      # layer zero is just the input layer
    linear_vals = {}
    
    # propogate through each layer of nn 
    for l in range(1, layers + 1):
        Z = np.matmul(weights["W" + str(l)], A) + weights["b" + str(l)]
        linear_vals["L" + str(l)] = Z

        # find which activation function is being used from the array activations
        if activations[l - 1] == "relu":        #relu is commonly used in middle layers more than others activaitons
            A = relu(Z)
        elif activations[l - 1] == "leakyRelu":
            A = leakyRelu(Z)
        elif activations[l - 1] == "sigmoid":
            A = sigmoid(Z)
        elif activations[l - 1] == "softmax":
            A = softmax(Z)
        elif activations[l - 1] == "tanh":
            A = tanh(Z)

        activ_vals["L" + str(l)] = A

    return activ_vals, linear_vals


# will need matrix calculus to understand how to solve
# for the gradients for the different types of activation functions
# in back propogation

# There are two types of gradients to solve for: derivatives with respect to the weights
# and derivatives with respect to A and Z

# note that Z = np.matmul(W, X) + b for each layer so the gradient of the cost function 
# with respect to W and b will always be 1/trainSize * (np.matmul(dZ^[L], A^[L].T)) and 
# 1/trainSize * (np.sum(dZ^[L], axis = 1, keepdims = True))for each layer [L] and where dZ^[L] is the gradient 
# of the cost function with respect to Z in layer [L] (derived using matrix calculus)


# the derivatives with respect to A and Z depend on the activation which is why
# we need separate gradient finding functions for each different type of activation

# back propogate, find the gradients needed to update weights
def back_propogation(weights, activations):
    layers = len(weights) // 2      # number of layers (excluding input layer)
    grads = {}          # we will store the gradients in this dicitonary

    # propogate through each layer of nn 
    for l in reversed(range(1, layers + 1)):
        A_new = weights["W" + l] 

        if activations[l] == 

    return grads