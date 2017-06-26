import numpy as np
from dataloader import *

np.set_printoptions(threshold = np.nan)
trainX, trainY, testX, testY = import_data()

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

clf = RandomForestClassifier( n_estimators = 10)
clf.fit (trainX, trainY)

yhat = clf.predict(testX)

print( metrics.classification_report(testY, yhat))
