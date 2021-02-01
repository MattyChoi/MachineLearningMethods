import numpy as np
import copy
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    bost = load_boston()
    Boston50 = copy.deepcopy(bost)
    Boston25 = copy.deepcopy(bost)
    bost50 = np.median(bost.target)
    bost25 = np.percentile(bost.target, 25)
    Boston50.target = np.where(Boston50.target >= bost50, 1, -1)
    Boston25.target = np.where(Boston25.target >= bost25, 1, -1)
    
    # number of features
    d = bost.data.shape[1]
    # number of max iterations
    iter = 1000
    # batch sizes
    m = [40, 200, bost.data.shape[0]]

    methodlist = [MySVM2(d, m[0], iter), MySVM2(d, m[1], iter), MySVM2(d, m[2], iter),
                LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=5000)]

    datalist = []
    methodName = []
    dataName = []

    for i in range(len(datalist)):
        for j in range(len(methodName)):
            errRates = 1 - cross_val_score(methodlist[j], datalist[i].data, datalist[i].target, cv=5, scoring="accuracy")
            print('Error rates for {} on {}:'.format(methodName[j], dataName[i]))
            for h in range(len(errRates)):
                print('Fold {}:  {}'.format(h + 1, errRates[h]))
            print('Mean:  {}'.format(np.mean(errRates)))
            print('Standard Deviation:  {}'.format(np.std(errRates)))


