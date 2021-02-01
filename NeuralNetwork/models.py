import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from costs import *
from initializations import *
from optimizations import *
from propogations import *

# returns the optimal weights of the neural network
# X = (number of featuers, number of examples), y = (1, number of examples)
# the neural network is already set to have mini batch learning, insert X.shape[1] for batch learning
def nn_model_mini_batch(X, y, layers_dims, activations, optimizer, initializer = "random", beta = 0.9, beta2 = 0.999, 
step_size = 0.01, mini_batch_size = 64, num_iterations = 5000, lambd = 0, print_cost = False):
    costs = []                # keep track of costs
    layers = len(activations)       # number of layers excluding the input layer
    trainSize = X.shape[1]

    # initialize weights
    weights = initial_weights(initializer, layers_dims)

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
            if mini_batch_size > trainSize:
                mini_batch_size = trainSize        # this is basically just regular batch learning
            minibatches = rand_mini_batches(X, y, size = mini_batch_size)
        else:
            minibatches = rand_mini_batches(X, y, size = 1)     # stochastic mini batch learning
        
        for minibatch in minibatches:
            mini_X, mini_y = minibatch
            
            # forward propogation to get the activation values
            prop_vals = forward_propogation(mini_X, weights, activations)
            
            # compute the cost so we can how well the nn is working
            if lambd:
                cost = cost(activations[-1], prop_vals["A" + str(layers)], mini_y, weights = weights, lambd = lambd)
            else:
                cost = cost(activations[-1], prop_vals["A" + str(layers)], mini_y)
            costs.append(cost)

            # backwards propogation to find the gradietns of the weights
            grads = back_propogation(weights, y, activations, prop_vals)
            
            # update weights through the given optimizer
            if optimizer == "grad_descent":
                weights = grad_descent(weights, grads, step_size = step_size)
            elif optimizer == "momentum":
                weights, velocity = grad_momentum(weights, grads, velocity, beta = beta, step_size = step_size)
            elif optimizer == "rms_prop":
                weights, velocity = rms_prop(weights, grads, velocity, beta = beta, step_size = step_size)
            elif optimizer == "adam":
                t += 1
                weights, velocity, sqr_veloc = adam_optim(weights, grads, velocity, sqr_veloc, t, beta1 = beta, beta2 = beta2, step_size = step_size)


    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.show()


    return weights

    
