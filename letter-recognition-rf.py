"""
    Program
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import pandas as pd

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Defining column names for the data set
COLUMNS = ["lettr", "x-bos", "y-box", "width", "high", "onpic", "x-bar", "y-bar",
	          "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]
# Distinguishing features from the label, also defining FEATURES and LABEL
FEATURES = ["x-bos", "y-box", "width", "high", "onpic", "x-bar", "y-bar", "x2bar",
	           "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]
LABEL = "lettr"
LABEL_VOCABULARY = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
                    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

LB = preprocessing.LabelEncoder()
LB.fit(LABEL_VOCABULARY)

FLAGS = None

def main(argv):
    """
		Main
    """
    print(FLAGS.max_depth)
    training_set = pd.read_csv("letter-recognition-training.csv", skipinitialspace=True,
		                             skiprows=0, names=COLUMNS)
    test_set = pd.read_csv("letter-recognition-test.csv", skipinitialspace=True,
                           skiprows=0, names=COLUMNS)
    prediction_set = pd.read_csv("letter-recognition-eval.csv", skipinitialspace=True,
		                               skiprows=0, names=COLUMNS)

    training_set[LABEL] = LB.transform(training_set[LABEL])
    test_set[LABEL] = LB.transform(test_set[LABEL])
    prediction_set[LABEL] = LB.transform(prediction_set[LABEL])

    # Training
    y = training_set[LABEL].values
    X = training_set[FEATURES].values

    # print(prediction_set)
    # print(prediction_set.as_matrix())

    clf = RandomForestClassifier(max_depth=FLAGS.max_depth, random_state=0)

    clf.fit(X, y)
    # print(clf.feature_importances_)

    # Evaluation
    y = test_set[LABEL].values
    X = test_set[FEATURES].values

    score = clf.score(X, y)
    cv_acc = cross_val_score(clf, X, y)

    print("Score : {}".format(score))
    print(cv_acc)

    # Prediction
    y = prediction_set[LABEL].values
    X = prediction_set[FEATURES].values

    y_p = clf.predict(X)

    print("TARGETS : {}".format(LB.inverse_transform(y)))
    print("PREDICTED : {}".format(LB.inverse_transform(y_p)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_depth',
        type=int,
        default=None,
        help='Maximum Depth of the Forest'
    )
    FLAGS, unparsed = parser.parse_known_args()
    main(argv=[sys.argv[0]] + unparsed)
