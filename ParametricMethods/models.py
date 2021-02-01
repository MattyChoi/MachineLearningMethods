import numpy as np
import sklearn 

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def my_mean(X):
    #Returns the means of the columns of X
    n = np.float64(X.shape[0])   #gets the size of the column
    return X.sum(axis=0) / n

def my_cov(X):
    #Returns the mle of the sample covariance matrix for X
    #biased since problem 2
    mu = np.ones(X.shape) * my_mean(X) #matrix whose rows are just the mean
    n = np.float64(X.shape[0])  # number of columns
    '''
    since the rows of X are x_t and since x_t - mu must be a column vector 
    and (x_t - mu)^T must be a row vector, we just transpose again
    note that the matrix multiplication automatically sums up the 
    (x_t - mu).T (x_t - mu) (column times row) for each t by construction
    '''
    diff = X - mu
    return (1 / n) * np.matmul(diff.T, diff)

class C_i:
    def __init__(self, label, X, k, diag):
        self.label = label
        self.num = X.shape[0]
        self.dim = X.shape[1]
        self.k = k          
        self.prior = 1. / np.float64(self.k)
        self.mean = np.zeros(self.dim)
        self.cov = np.eye(self.dim)
        self.diag = diag

        #set parameters now using given arguments
        self.cov = my_cov(X)    #calculates mle covariance
        self.mean = my_mean(X)    #calculates mle the mean
        self.prior = self.num / np.float64(self.k)    #calculates mle prior prob

        if self.diag:   #creates a diagonal matrix by making 
        #only the diagonal entries nonzero
            self.cov = (np.eye(self.dim)) * self.cov

        #calculates nonzero determinant of covariance for discriminant
        while np.linalg.det(self.cov) == 0:    #cannot have zero determinant
            self.cov += 0.0001*np.eye(self.dim)
            #self.cov += np.random.normal(0, 0.001, self.cov.shape)
        detCov = np.linalg.det(self.cov)

        #calculates inverse of covariance for discriminant
        invCov = np.linalg.inv(self.cov)

        #coefficients of the components of the discriminant
        #universal for any value of X we predict on in the predict method
        if not self.diag:
            self.W_i = (-1./2.) * invCov
            self.w_i = np.matmul(invCov, self.mean.T)
            self.w_i0 = (-1./2.) * np.matmul(np.matmul(self.mean, invCov), self.mean.T) - (1./ 2.) * np.log(detCov) + np.log(self.prior)

    def discrim(self, X):
        discrim = 0
        if self.diag:
            for i in range(self.dim):
                discrim += ((X[i]-self.mean[i])/self.cov[i][i])**2
            discrim = (-1./2.)*discrim + np.log(self.prior)
        else: 
            discrim = np.matmul(np.matmul(X, self.W_i), X.T) + self.w_i.T.dot(X.T) + self.w_i0
        return discrim

class MultiGaussClassify:
    def __init__(self, k, d, diag=False):
        self.k = k      #number of classes
        self.dim = d       #number of features
        self.diag = diag    #diagonal or not
        self.classes = []       #list of classes

    def fit(self, X, y):
        self.classes = []      #resets list of classes
        labels = np.unique(y)      #creates a list of the different types of labels
        self.k = len(labels)
        #self.dim = X.shape[1]
        for label in labels:
            index = np.where(y == label)[0]    #index list of 
            X_i = X[index]    #extracts data that correspond to desired label
            c_i = C_i(label, X_i, self.k, self.diag)         #create a class with label label
            self.classes.append(c_i)        #add each class to list of classes

    def predict(self, X):
        ypred = []
        for i in range(X.shape[0]):
            prob = []
            for c_i in self.classes:
                disc = c_i.discrim(X[i])
                prob.append(disc)
            high = np.argmax(prob)
            ypred.append(self.classes[high].label) # Return the label for the class with the highest prob

        return np.array(ypred)

    def get_params(self, deep=True):
        return {"k": self.k, "d": self.dim, "diag": self.diag}

