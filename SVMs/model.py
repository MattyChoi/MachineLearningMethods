import numpy as np
from numpy import linalg as la
import sklearn

from sklearn.metrics import accuracy_score

class MySVM2:

    # Init functions
    def __init__(self, d, m, iter):
        self.d = d
        self.m = m
        self.w = 0.02 * np.random.random_sample((self.d+1,)) - 0.01
        self.iter = iter
        self.epsilon = 0.00001
        self.eta = 0.001

    # Fit the dataset
    def fit(self, X, y):
        # Convert data to NumPy array as precaution
        X = np.array(X)
        y = np.array(y)

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        # minibatch for mini-batch stochastic gradient descent
        batch = np.array(np.random.choice(X.shape[0], self.m))
        X = X[batch]
        y = y[batch]

        # Normalize each feature to account for scale and center
        X = (X - self.mean) / self.std

        # Make a prediction with current W and find loss
        prevVecW = self.w[1:]
        prevW0 = self.w[0]
        # prevHyp is the activation function
        prevProd = np.multiply(y, np.matmul(X, prevVecW) + prevW0)
        prevHyp = np.where(prevProd < 1, 1, 0)
        # prevLoss calculates the loss from the original
        prevLoss = 2.5 * la.norm(prevVecW)**2 + (np.matmul(1 - prevProd, prevHyp))/X.shape[0]


        # Iterate till convergence or iter
        for it in range(self.iter):
            # Calculate the gradient and hence the delta changes
            gradVecW = 5.0 * (prevVecW) - (X.T.dot(np.multiply(prevHyp, y)))/X.shape[0]
            gradW0 = - np.matmul(prevHyp, y)/y.shape[0]
            deltaW = (self.eta*gradVecW)
            deltaW0 = (self.eta*gradW0)

            # Calculate the new W and loss
            newVecW = prevVecW - deltaW      #grad descent part
            newW0 = prevW0 - deltaW0
            newProd = np.multiply(y, np.matmul(X, newVecW) + newW0)
            newHyp = np.where(newProd < 1, 1, 0)
            newLoss = 2.5 *la.norm(newVecW)**2 + (np.matmul(1 - newProd, newHyp))/X.shape[0]
            # Calculate the new W and loss
            newVecW = prevVecW - deltaW      #grad descent part
            newW0 = prevW0 - deltaW0
            newProd = np.array([y[i] * (np.matmul(X[i], newVecW) + newW0) for i in range(y.shape[0])])
            newHyp = np.where(newProd < 1, 1, 0)
            newLoss = 2.5 *(newVecW.dot(newVecW)) + np.sum(np.multiply((1 - newProd), newHyp))/X.shape[0]
            
            # Check for convergence
            if abs(prevLoss-newLoss) < self.epsilon:
                break

            # Assign the calculated weights, predictions, and loss as old
            prevHyp = newHyp
            prevLoss = newLoss
            prevProd = newProd
            prevVecW = newVecW
            prevW0 = newW0

        self.w = np.insert(newVecW, 0, newW0)
    
    # Predict the value of new dataPoints
    def predict(self, X):
        XNorm = (X - self.mean) / self.std
        XFinal = np.insert(XNorm, 0, np.ones(X.shape[0]), axis=1)
        return 2 * ((np.matmul(XFinal, self.w) > 0).astype(int)) - 1

    def get_params(self, deep=True):
        return {"d": self.d, "iter": self.iter, "m": self.m}