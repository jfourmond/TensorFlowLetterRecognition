"""
    Random Forest model applied to the Letter Recognition Dataset
    (Using scikit-learn)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import pandas as pd

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

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
    start = time.time()

    print("NUMBER OF TREE : {}".format(FLAGS.n_trees))
    print("MAX DEPTH : {}".format(FLAGS.max_depth))
    print("SEED : {}".format(FLAGS.random_state))

    print("RESULTS FILE : {}".format(FLAGS.results_file))

    N_TREES = FLAGS.n_trees
    MAX_DEPTH = FLAGS.max_depth
    RANDOM_STATE = FLAGS.random_state

    RESULTS_FILE = FLAGS.results_file

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

    clf = RandomForestClassifier(
        n_estimators=N_TREES,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE)

    clf.fit(X, y)

    # Evaluation
    y = test_set[LABEL].values
    X = test_set[FEATURES].values

    score = clf.score(X, y)

    print("Score : {}".format(score))

    # Prediction
    y = prediction_set[LABEL].values
    X = prediction_set[FEATURES].values

    y_p = clf.predict(X)
    s_p = accuracy_score(y, y_p)

    print("TARGETS : {}".format(LB.inverse_transform(y)))
    print("PREDICTED : {}".format(LB.inverse_transform(y_p)))
    print("ACCURACY PREDICTION : {}".format(s_p))

    end = time.time() - start
    print("Execution time :  {:.4f} seconds\n\n".format(end))

    if RESULTS_FILE:
        resultsFile = open(RESULTS_FILE, "a")
        # n_trees, Accuracy, Time
        resultsFile.write("{}, {}, {}\n".format(N_TREES, score, end))
        resultsFile.close

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_depth',
        type=int,
        default=None,
        help='Maximum Depth of the tree'
    )
    parser.add_argument(
        '--n_trees',
        type=int,
        default=10,
        help='Number of trees in the forest'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=None,
        help="Seed used by the random number generator"
    )
    parser.add_argument(
        '--results_file',
        type=str,
        default=None,                        #"letter-recognition-rf.csv"
        help='File where results will be stored'
    )
    FLAGS, unparsed = parser.parse_known_args()
    main(argv=[sys.argv[0]] + unparsed)
