import numpy as np

# initialize weights as zeros given a list containing the size of each layer
def zero_initial(layers_dims):
    weights = {}
    
    for l in range(1, len(layers_dims)):
        weights["W" + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        weights["b" + str(l)] = np.zeros((layers_dims[l], 1))
    return weights


# initialize weights with random numbers to break symmetry
# break symmetry = if weights initialized the same, the units in each hidden layer
# will be identical
def rand_initial(layers_dims):
    weights = {}

    for l in range(1, len(layers_dims)):
        weights["W" + str(l)] = np.random.randn((layers_dims[l], layers_dims[l-1]))   # random with normal distribution
        #weights["W" + str(l)] = np.random.randn((layers_dims[l], layers_dims[l-1]))   # random with uniform distribution
        weights["b" + str(l)] = np.zeros((layers_dims[l], 1))
    return weights


# this is for relu activation
# helps with vanishing or exploding gradient descents which slows down optimization
def he_initial(layers_dims):
    weights = {}

    for l in range(1, len(layers_dims)):
        weights["W" + str(l)] = np.random.randn((layers_dims[l], layers_dims[l-1])) * (2./layers_dims[l-1])**0.5
        weights["b" + str(l)] = np.zeros((layers_dims[l], 1))
    return weights
    

# this is for tanh activation
# helps with vanishing or exploding gradient descents which slows down optimization
def xavier_initial(layers_dims):
    weights = {}

    for l in range(1, len(layers_dims)):
        weights["W" + str(l)] = np.random.randn((layers_dims[l], layers_dims[l-1])) * (1./layers_dims[l-1])**0.5
        weights["b" + str(l)] = np.zeros((layers_dims[l], 1))
    return weights


# velocity initialization for gradient descent with momentum
def initial_veloc(weights):
    layers = len(weights)//2
    # store velocity as zeros
    velocity = {}

    for l in range(1, layers + 1):
        velocity["dW" + str(l)] = np.zeros(weights["W" + l].shape)
        velocity["db" + str(l)] = np.zeros(weights["b" + l].shape)
    return velocity


# velocity and sqaured velocity initialization for adam optimization
def initial_adam(weights):
    layers = len(weights)//2
    # store velocity as zeros
    velocity = {}
    sqr_veloc = {}

    for l in range(1, layers + 1):
        velocity["dW" + str(l)] = np.zeros(weights["W" + l].shape)
        velocity["db" + str(l)] = np.zeros(weights["b" + l].shape)
        sqr_veloc["dW" + str(l)] = np.zeros(weights["W" + l].shape)
        sqr_veloc["db" + str(l)] = np.zeros(weights["b" + l].shape)
    return velocity, sqr_veloc

def initial_weights(initializer, layers_dims):
    if initializer == "random":
        return rand_initial(layers_dims)
    elif initializer == "zero":       # not recommended if using hidden layers
        return zero_initial(layers_dims)
    elif initializer == "he":
        return he_initial(layers_dims)
    elif initializer == "xavier":
        return xavier_initial(layers_dims)