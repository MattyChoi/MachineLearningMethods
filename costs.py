import numpy as np

# p = prediction, y = actual

# mean square error
def meanSquareError(p, y):
    cost = 0.5 * np.sum((p - y)**2)
    return cost


# cross_entropy for logistic regression
# y = (1, # of examples)
def cross_entropy(p, y):
    cost = -np.sum(np.multiply(y, np.log(p)) + np.multiply((1-y), np.log(1-p)))/y.shape[1]
    cost = np.squeeze(cost) #double check cost is a number, not array
    return cost


# cross_entropy with l2 regularization
def reg_cross_entropy(p, y, weights, lambd):
    cost = -np.sum(np.multiply(y, np.log(p)) + np.multiply((1-y), np.log(1-p)))
    for w in weights:
        cost += lambd/2 * np.sum(w**2)
    cost = np.squeeze(cost/y.shape[1]) #double check cost is a number, not array
    return cost


# cross_entropy for softmax regression
# y = (# of classes, # of examples)
def softmax_cross_entropy(p, y):
    cost = -np.sum(np.multiply(y, np.log(p)))/ y.shape[1]
    cost = np.squeeze(cost) #double check cost is a number, not array
    return cost


# softmax_cross_entropy with l2 regularization
def reg_softmax_cross_entropy(p, y, weights, lambd):
    cost = -np.sum(np.multiply(y, np.log(p)))
    for w in weights:
        cost += lambd/2 * np.sum(w**2)
    cost = np.squeeze(cost/ y.shape[1]) #double check cost is a number, not array
    return cost


# cross_entropy for tanh activation
# y = (# of classes, # of examples)
def tanh_cross_entropy(p, y):
    cost = (-0.5 * np.sum(np.multiply(1+y, np.log(1+p)) + np.multiply((1-y), np.log(1-p))) + np.log(2))/y.shape[1]
    cost = np.squeeze(cost) #double check cost is a number, not array
    return cost


# tanh_cross_entropy with l2 regularization
def reg_tanh_cross_entropy(p, y, weights, lambd):
    cost = -0.5 * np.sum(np.multiply(1+y, np.log(1+p)) + np.multiply((1-y), np.log(1-p))) + np.log(2)
    for w in weights:
        cost += lambd/2 * np.sum(w**2)
    cost = np.squeeze(cost/y.shape[1]) #double check cost is a number, not array
    return cost