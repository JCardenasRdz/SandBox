import numpy as np
from sklearn import metrics

def _print_info(trainX, testX):
    print('x_train shape:', trainX.shape)
    print(trainX.shape[0], 'train samples')
    print(testX.shape[0], 'test samples')
    print(40*'=')

def fit_RandomForest(trainX, trainY, testX, testY, n_estimators = 10):
    from sklearn.ensemble import RandomForestClassifier

    _print_info(trainX, testX)

    clf = RandomForestClassifier( n_estimators = n_estimators, random_state= 123, n_jobs = -1)
    clf.fit (trainX, trainY)
    yhat = clf.predict(testX)

    report = metrics.classification_report(testY, yhat)
    print( report)
    return clf

def fit_ExtraTrees(trainX, trainY, testX, testY, n_estimators = 10, max_depth = None):
    from sklearn.ensemble import ExtraTreesClassifier
    import matplotlib.pyplot as plt

    _print_info(trainX, testX)

    # fit
    clf = ExtraTreesClassifier( n_estimators = n_estimators,
                                random_state= 123,
                                max_depth = max_depth,
                                n_jobs = -1)
    clf.fit (trainX, trainY)
    # predict
    yhat = clf.predict(testX)

    # Peformance for each classifier Tree
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                                                        axis=0)
    indices = np.argsort(importances)[::-1]

    # Plot the feature importances of the forest
    #plt.figure(figsize = (15,5))
    #plt.title("Feature importances")
    #plt.bar(range(trainX.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
    #plt.xticks(range(trainX.shape[1]), indices)
    #plt.xlim([-1, trainX.shape[1]])
    #plt.show()

    report = metrics.classification_report(testY, yhat)
    print( report)
    return clf

def fit_Logistic_Regression(trainX, trainY, testX, testY, C=1.0, fit_intercept=True):
    from sklearn.linear_model import LogisticRegression

    _print_info(trainX, testX)

    lgc = LogisticRegression(C = C, fit_intercept = fit_intercept)
    lgc.fit(trainX, trainY)
    yhat = lgc.predict(testX)

    report = metrics.classification_report(testY, yhat)
    print( report)
    return report
