import numpy as np

# famous gradient descent funciton to update the weights
def grad_descent(weights, grads, step_size = 0.01):
    # find the number of layers excluding the input layer
    # (this would be layers_dims - 1)
    layers = len(weights)//2

    for l in range(1, layers + 1):
        weights["W" + str(l)] -= step_size * grads["dW" + str(l)]
        weights["b" + str(l)] -= step_size * grads["db" + str(l)]
    
    return weights


# implement stochastic gradient descent or mini batch gradient descent
# using mini batches (in this function, stochastic is size = 1)
# common sizes are 64, 128, 256, 512
def rand_mini_batches(X, y, size = 64):
    trainSize = X.shape[1]      # number of training examples
    mini_batches = []

    # shuffle the training data and classes the same way
    permute = list(np.random.permutation(trainSize))
    shuffled_X = X[:, permute]
    shuffled_y = y[:, permute]

    # add to mini_batches
    num_batches = trainSize // size     # number of minibatches (possibly one more)
    for num in range(num_batches):
        mini_batch_X = shuffled_X[:, num * size: (num + 1) * size]
        mini_batch_y = shuffled_y[:, num * size: (num + 1) * size]
        mini_batches.append((mini_batch_X, mini_batch_y))

    # add remaining minibatch if exists
    if trainSize % size != 0:
        mini_batch_X = shuffled_X[:, num_batches * size:]
        mini_batch_y = shuffled_y[:, num_batches * size:]
        mini_batches.append((mini_batch_X, mini_batch_y))

    return mini_batches


# gradient descent with momentum, uses momentum to reduce variance
# of path of mini batch learning (reduce oscillations as mini batch
# learning does not go straight to point)
def grad_momentum(weights, grads, velocity, beta = 0.9, step_size = 0.01):
    # find the number of layers excluding the input layer
    # (this would be layers_dims - 1)
    layers = len(weights)//2

    for l in range(1, layers + 1):
        # update velocities first
        velocity["dW" + str(l)] = beta * velocity["dW" + str(l)] + (1-beta) * grads["dW" + str(l)]
        velocity["db" + str(l)] = beta * velocity["db" + str(l)] + (1-beta) * grads["db" + str(l)]

        # update weights now using new velocities
        weights["W" + str(l)] -= step_size * velocity["dW" + str(l)]
        weights["b" + str(l)] -= step_size * velocity["db" + str(l)]
    
    return weights

# similar to gradient descent with momentum except it slows down learning for weights
# that are more optimized and speeds up learning for weights that need more 
# need more optimization
def rms_prop(weights, grads, velocity, beta = 0.9, step_size = 0.01):
    # find the number of layers excluding the input layer
    # (this would be layers_dims - 1)
    layers = len(weights)//2

    for l in range(1, layers + 1):
        # update velocities first
        # makes small velocity smaller and big velocity bigger
        velocity["dW" + str(l)] = beta * velocity["dW" + str(l)] + (1-beta) * grads["dW" + str(l)]**2
        velocity["db" + str(l)] = beta * velocity["db" + str(l)] + (1-beta) * grads["db" + str(l)]**2

        # update weights now using new velocities
        # divide by velocity square rooted so learning sped up by small velocity and learning slowed
        # down for big velocity
        weights["W" + str(l)] -= step_size * grads["dW" + str(l)] / velocity["dW" + str(l)]**0.5
        weights["b" + str(l)] -= step_size * grads["db" + str(l)] / velocity["db" + str(l)]**0.5
    
    return weights, velocity


# combination of gradient momentum and rms prop
def adam_optim(weights, grads, velocity, sqr_veloc, t, beta1 = 0.9, beta2 = 0.999, step_size = 0.01):
    # find the number of layers excluding the input layer
    # (this would be layers_dims - 1)
    epsilon = 1e-8
    layers = len(weights)//2

    for l in range(1, layers + 1):
        # update velocities and square velocities first
        # uses velocity and squared velocity
        velocity["dW" + str(l)] = beta1 * velocity["dW" + str(l)] + (1-beta1) * grads["dW" + str(l)]
        velocity["db" + str(l)] = beta1 * velocity["db" + str(l)] + (1-beta1) * grads["db" + str(l)]
        sqr_veloc["dW" + str(l)] = beta2 * sqr_veloc["dW" + str(l)] + (1-beta2) * grads["dW" + str(l)]**2
        sqr_veloc["db" + str(l)] = beta2 * sqr_veloc["db" + str(l)] + (1-beta2) * grads["db" + str(l)]**2

        # bias-corrections
        corr_vel_w = velocity["dW" + str(l)] / (1 - beta1**t)
        corr_vel_b = velocity["db" + str(l)] / (1 - beta1**t)
        corr_sqr_w = sqr_veloc["dW" + str(l)] / (1 - beta2**t)
        corr_sqr_b = sqr_veloc["db" + str(l)] / (1 - beta2**t)

        # update weights now
        weights["W" + str(l)] -= step_size * corr_vel_w / (corr_vel_b**0.5 + epsilon)
        weights["b" + str(l)] -= step_size * corr_sqr_w / (corr_sqr_b**0.5 + epsilon)
    
    return weights, velocity, sqr_veloc