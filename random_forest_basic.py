"""
This program attempts to experiment if a machine learning (RandomForest) can solve the
following problem:

Given a fixed array of 5 elements consisting of at most ONE of each element
 from (1, 2, 3, 4, 5) and ONE OR MORE (0), replace the 0s with appropriate values so 
 that the final array has exactly ONE of each (1, 2, 3, 4, 5).

- Generate N examples of combinations configurations with elements of [1, 2, 3, 4, 5]
  as answers, and randomly replace some of the elements as 0s.
- For instance, one training example is [(0, 3, 5, 4, 0), (2, 3, 5, 4, 1)].
- There can be multiple same input mapping to different output, i.e. [(0, 3, 5, 4, 0),
  (2, 3, 5, 4, 1)] and [(0, 3, 5, 4, 0), (1, 3, 5, 4, 2)] can be both present as two
  separate training instances.
- Separate the training data set 10 fold, shuffled, and train using a
  RandomForestClassifier from Scikit-Learn.
- A correct output is defined as the final configuration array has exactly ONE element
  from (1, 2, 3, 4, 5). So (2, 4, 4, 5, 1) is not valid.

Observation:
RandomForestClassifier fails to learn this problem. Given incraesing number of examples
from 1000, 10000, 50000, 100000, the accuracy stays the same. So increasing examples does
not help the model to learn the problem, so there is something fundamentally lacking in
the model to learn this problem.

See stackoverflow question here:
http://stackoverflow.com/questions/33734567/why-does-the-model-fail-to-learn-this-game-of-filling-up-integers
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
    Y = np.array(Y)

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