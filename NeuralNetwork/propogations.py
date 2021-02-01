import numpy as np
from activations import *


def forward_activation(activation, Z):
    # find which activation function is being used from the array activations
    if activation == "relu":        #relu is commonly used in middle layers more than others activaitons
        return relu(Z)
    elif activation == "leaky":
        return leakyRelu(Z)
    elif activation == "sigmoid":
        return sigmoid(Z)
    elif activation == "softmax":
        return softmax(Z)
    elif activation == "tanh":
        return tanh(Z)
    return Z       # assume it is a linear activation function


# forward propogate, predict using the current weights
def forward_propogation(X, weights, activations):
    layers = len(weights) // 2      # number of layers (excluding input layer)
    A = X       # A will be activation function

    prop_vals = {"A0": A}      # layer zero is just the input layer
    
    # propogate through each layer of nn 
    for l in range(1, layers + 1):
        # Z is always 
        Z = np.matmul(weights["W" + str(l)], A) + weights["b" + str(l)]
        prop_vals["Z" + str(l)] = Z

        # put Z through activation function for this layer to get activation value
        prop_vals["A" + str(l)] = forward_activation(activations[l - 1], Z)

    return prop_vals


# will need matrix calculus to understand how to solve
# for the gradients for the different types of activation functions
# in back propogation

# There are two types of gradients to solve for: derivatives with respect to the weights
# and derivatives with respect to A and Z

# note that Z = np.matmul(W, A_prev) + b for each layer so the gradient of the cost function 
# with respect to W and b will always be 1/trainSize * (np.matmul(dZ^[L], A^[L].T)) and 
# 1/trainSize * (np.sum(dZ^[L], axis = 1, keepdims = True)) for each layer [L] and where dZ^[L] is the gradient 
# of the cost function with respect to Z in layer [L] and A_prev is the activation value
# of the previous layer (derived using matrix calculus)
def grad_linear(dZ, A_prev):
    dW = np.matmul(dZ, A_prev)
    db = np.sum(dZ, axis = 1, keepdims=True)
    return dW, db


# the derivatives with respect to A and Z depend on the activation which is why
# we need separate gradient finding functions for each different type of activation
# we do not use softmax in the middle layers
def grad_activ(activation, dA, Z):
    # to get dZ from dA, all we must do is find the derivative of A with respect to Z
    # which is just the derivative of the activaiton function and element-wise multiply with dA
    # let g(x) be the activation function
    if activation == "relu":        #relu is commonly used in middle layers more than others activaitons
        # the derivative g'(Z) of the relu function g(Z) is simply 1 when Z > 0 and 0 otherwise
        # actually the derivative is nonexistent at z = 0 but it happens so rarely so we ignore it here
        return dA * np.int64(Z > 0)
    elif activation == "leaky":
        # similarly derivative g'(Z) of the leakyrelu function g(Z) is simply 1 when Z > 0 and 0.01 otherwise
        # (or whatever constant you used for the leakyrelu funciton)
        return dA * (np.int64(Z > 0) + 0.01 * np.int64(Z <= 0))
    elif activation == "sigmoid":
        # derivative g'(Z) of the sigmoid function g(Z) is g'(Z) = g(Z)(1 - g(Z))
        g = sigmoid(Z)
        return dA * g * (1-g)
    elif activation == "tanh":
        # derivative g'(Z) of the sigmoid function g(Z) is g'(Z) = g(Z)(1 - g(Z))
        g = tanh(Z)
        return dA * (1-g**2)
    else:
        return dA           # if activation is linear, then dA = dZ or 



# this outputs the result of the gradient of the cost function with respect 
# to the output layer activation value
# note that we do not divide by the number of training data, we do that in grad_linear
'''
def grad_cost(activation, A_L, y):
    # the cost function depends on the type of activation function used in the 
    # output layer
     # note that relu or leakyrelu are not in the choices
    # that's because using a relu or leakyrelu activaiton function
    # for your output does not make much sense
    if activation == "sigmoid":
        # derivative of cross_entropy with respect to A_L (A_L = p)
        # -y / A_L + (1 - y)/(1-A_L)
        # multiply by derivative of sigmoid function A_L(1-A_L) to get
        return A_L - y      # derivative of cost function with respect to Z
    elif activation == "softmax":
        # the softmax function is different from the other activaiton functions
        # since it is not linear so the derivative is a matrix, not a scalar
        # however, since we are also vectorizing, this derivative would be a three dimensional
        # matrix which is hard to implement
        return A_L - y      # derivative of cost function with respect to Z
    elif activation == "tanh":
        # derivative of tanh of cross_entropy with respect to A_L is
        # -0.5 * ( (1 + y)/(1+A_L) - (1 - y)/(1-A_L) )
        # multiply by derivative of tanh ufnction with respect to z
    return A_L - y      # derivative of mean square error
'''

# back propogate, find the gradients needed to update weights
def back_propogation(weights, y, activations, prop_vals):
    layers = len(weights) // 2      # number of layers (excluding input layer)
    trainSize = y.shape[1]
    grads = {}          # we will store the gradients in this dicitonary

    # the 1 / trainsize from cost function is implemeted here
    # the derivative of all the cost functions with respect to Z are (A_L - y) / # of training examples
    dZ = (prop_vals["A" + str(layers)] - y) / trainSize        
    dW, db = grad_linear(dZ, prop_vals["A" + str(layers - 1)])
    
    # store the gradiesnt of the weights in grads
    # note that we do not need dA nor dZ
    grads["dW" + layers] = dW
    grads["db" + layers] = db

    # propogate through each layer of nn 
    for l in reversed(range(1, layers)):
        dA = np.matmul(weights["W" + str(l)].T, dZ)     # since Z = np.matmul(W, A_prev) + b
        dZ = grad_activ(activations[l-1], dA, prop_vals["Z" + str(l)]) / trainSize 
        dW, db = grad_linear(dZ, prop_vals["A" + str(l-1)])

        # store the gradients of the weights
        grads["dW" + layers] = dW
        grads["db" + layers] = db

    return grads