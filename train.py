import numpy as np
import random

def my_cross_val(method, X, y, k):
    Xtrain = []   #train on this set of points
    Xtest = []    #test on this set of points
    ytrain = []   #train on this set of points
    ytest = []    #test on this set of points
    errRate = []
    errs = 0       #number of errors
    for i in range(k):    #cycles around to make sure each of the k folds is tested on
        for j in range(len(X)):   #goes through each data point
            if (j % k) != i:    #(k - 1) parts of the k parts is added to the training set
                Xtrain.append(X[j])   
                ytrain.append(y[j])
            else:     #adds every kth data point after the ith data point 
            #(this way, if it doesn't divide evenly, each fold has about the same number of data points)
                Xtest.append(X[j])
                ytest.append(y[j])
        method.fit(Xtrain, ytrain)    # fits the training data into the given model
        #The next for loop goes through each data point in the validation set Xtest
        #and dividing it by how many times it could've occurred (errRate)
        for h in range(len(Xtest)): 
            newtest = np.array(Xtest[h]).reshape(-1, X.shape[1])  #X.shape[1] measure the number of features
            if method.predict(newtest) != ytest[h]:
                errs += 1
        #Append to a list and reset vars
        errRate.append(errs/len(Xtest))  
        errs = 0
        Xtrain = []   #resets the training and testing data
        Xtest = []
        ytrain = []
        ytest = []
    return errRate

# trains on random samples of fraction pi of the data to test on
def my_train_test(method, X, y, pi, k):
    Xtrain = []   #train on this set of points
    Xtest = []    #test on this set of points
    ytrain = []   #train on this set of points
    ytest = []    #test on this set of points
    errRate = []
    errs = 0       #number of errors
    for i in range(k):    #takes a random sampling of size a fraction pi of the data k times
        rand = random.sample(range(len(y)), int(len(y) * pi))    #creates a random sample of indexes for training
        for j in range(len(y)):   #goes through each data point
            if j in rand:    #adds all the data points whose indexes are in rand
                Xtrain.append(X[j])   
                ytrain.append(y[j])
            else:     #adds the data points that aren't
                Xtest.append(X[j])
                ytest.append(y[j])
        method.fit(Xtrain, ytrain)    # fits the training data into the given model
        #The next for loop goes through each data point in the validation set Xtest
        #and dividing it by how many times it could've occurred (errRate)
        for h in range(len(Xtest)): 
            newtest = np.array(Xtest[h]).reshape(-1, X.shape[1])  #X.shape[1] measure the number of features
            if method.predict(newtest) != ytest[h]:
                errs += 1
        #Append to a list and reset vars
        errRate.append(errs/len(Xtest))   
        errs = 0
        Xtrain = []   #resets the training and testing data
        Xtest = []
        ytrain = []
        ytest = []
    return errRate