"""
This program attempts to experiment if a machine learning (RandomForest) can solve the
following problem:

Given a fixed array of 5 elements consisting of at most ONE of each element
 from (1, 2, 3, 4, 5) and ONE OR MORE (0), replace the 0s with appropriate values so 
 that the final array has exactly ONE of each (1, 2, 3, 4, 5).

This is the second attempt, I generated a feature that represents the residual value to
fill in. That is, if [3, 4, 5, 0, 0] is provided, then the remaining residual is 3. This
is an attempt to provide the model some higher level feature and see if it learn take
this hint and learn the program better.

The observation is that the model cannot learn this problem even given this feature.
"""
import csv
import sys

import numpy as np
from sklearn.cross_validation import *
from sklearn.ensemble import *
from sklearn.metrics import *

# Returns true if the result is a valid configuration.
def valid_output(row):
    return len(set(row)) == len(row)

# Returns the zero one loss based on validity.
def cal_validity_score(preds):
    n = len(preds)
    valid_preds = [pred for pred in preds if valid_output(pred)]
    return len(valid_preds), len(valid_preds) / n

if __name__ == '__main__':
    reader = csv.reader(open(sys.argv[1], 'r'))
    answer_len = int(reader.__next__()[0])
    print(answer_len)

    X = []
    Y = []
    for row in reader:
        row = [int(i) for i in row]
        features_vec = row[:answer_len]
        target_vec = row[answer_len:]
        X.append(features_vec)
        Y.append(target_vec)
        # print '%s => %s' % (features_vec, target_vec)
    X = np.array(X)
    sum_column = np.array([15 - np.sum(X[i, :]) for i in range(X.shape[0])]).reshape((X.shape[0], 1))
    X = np.hstack((X, sum_column))
    Y = np.array(Y)
    
    print(X.shape)

    validity_scores = []
    for k, (train, cv) in enumerate(KFold(len(Y), n_folds = 10, shuffle = True, random_state = 144)):
        X_train = X[train, :]
        Y_train = Y[train, :]
        X_cv = X[cv, :]
        Y_cv = Y[cv, :]
        
        clf = RandomForestClassifier()
        clf.fit(X_train, Y_train)
        Y_cv_pred = clf.predict(X_cv)
        Y_cv_pred = Y_cv_pred.astype(int)

        if k == 0:
            for i in range(len(Y_cv_pred)):
                print('%s => %s | %s' % (X_cv[i], Y_cv_pred[i], valid_output(Y_cv_pred[i])))
        n_valid_preds, validity_score = cal_validity_score(Y_cv_pred)
        
        validity_scores.append(validity_score)
        
        print('validity score %d/%d : %f' % (n_valid_preds, len(Y_cv_pred), validity_score))
        print('cv len: %d' % (len(Y_cv_pred)))
    print('%f | %s' % (np.mean(validity_scores), validity_scores))