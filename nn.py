import numpy as np
import tensorflow as tf

from costs import *
from initializations import *
from optimizations import *

# returns the optimal weights of the neural network
# X = (number of featuers, number of examples), y = (1, number of examples)
def nn_model_mini_batch(X, y, layers_dims, activations, optimizer, initializer = "random",
 beta = 0.9, beta2 = 0.999, step_size = 0.01, mini_batch_size = 64, num_iterations = 5000, print_cost = False):
    # costs = []                # keep track of costs
    layers = len(activations)       # number of layers excluding the input layer

    # initialize weights
    if initializer == "random":
        weights = rand_initial(layers_dims)
    elif initializer = "zero":       # not recommended if using hidden layers
        weights = zero_initial(layers_dims)
    elif initializer = "he":
        weights = he_initial(layers_dims)
    elif initializer = "xavier":
        weights = xavier_initial(layers_dims)


    # Initialize the optimizer
    if optimizer == "momentum" or optimizer == "rms_prop":
        velocity = initial_veloc(weights)
    elif optimizer == "adam":
        velocity, sqr_veloc = initial_adam(weights)
        t = 0

    # start the modeling
    for iter in range(num_iterations):
        # if batch size is negative for some reason, do stochastic mini batches
        if mini_batch_size > 1:
            if mini_batch_size > X.shape[1]:
                mini_batch_size = X.shape[1]
            minibatches = rand_mini_batches(X, y, size = mini_batch_size)
        else:
            minibatches = rand_mini_batches(X, y, size = 1)
        
        for minibatch in minibatches:
            mini_X, mini_y = minibatch
            
            forward_propogation(X, weights, activations)

            back_propogation(weights, activations)
            
            # last activation determines what cost function to use
            elif activations[l - 1] == "sigmoid":
                A = sigmoid(Z)
            elif activations[l - 1] == "softmax":
                A = softmax(Z)
            elif activations[l - 1] == "tanh":
                A = tanh(Z)
            if activations[-1] == "relu":        # relu and leakyrelu rarely used for output layer
                A = relu(Z)
            elif activations[-1] == "leakyRelu":
                A = leakyRelu(Z)

            if optimizer == "grad_descent":
                weights = grad_descent(weights, grads, step_size = step_size)
            elif optimizer == "momentum":
                weights, velocity = grad_momentum(weights, grads, velocity, beta = beta, step_size = step_size)
            elif optimizer == "rms_prop":
                weights, velocity = rms_prop(weights, grads, velocity, beta = beta, step_size = step_size)
            elif optimizer == "adam":
                t += 1
                weights, velocity, sqr_veloc = adam_optim(weights, grads, velocity, sqr_veloc, t, beta1 = beta, beta2 = 0.999, step_size = 0.01)






    return weights