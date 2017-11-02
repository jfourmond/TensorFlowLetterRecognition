"""
    Program
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf

# Setting logging verbosity to INFO for more detailed log output
tf.logging.set_verbosity(tf.logging.INFO)

# Defining column names for the data set
COLUMNS = ["lettr", "x-bos", "y-box", "width", "high", "onpic", "x-bar", "y-bar",
	          "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]
# Distinguishing features from the label, also defining FEATURES and LABEL
FEATURES = ["x-bos", "y-box", "width", "high", "onpic", "x-bar", "y-bar", "x2bar",
	           "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]
LABEL = "lettr"
LABEL_VOCABULARY = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
                    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

HIDDEN_UNITS = [100, 100]

def get_input_fn(data_set, num_epochs=None, shuffle=True):
    """
    	Building the input_fn
    	Two arguments :
    		- num_epochs : controls the number of epochs to iterate over data.
    			For training, set this to None, so the input_fn keeps returning data until the
    			required number of train steps is reached.
    			For evalute and predict, set this to 1, so the input_fn will iterate over the
    			data once and then raise OutOfRangeError.
    			That error will signal the Estimator to stop evaluate or predict.
    		- shuffle : whether to shuffle the data.
    			For evaluate and predict, set this to False, so the input_fn iterates over the data
    			sequentially. For train, set this to True.
    """
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
        y=pd.Series(data_set[LABEL].values),
        num_epochs=num_epochs,
        shuffle=shuffle)

def main(_):
    """
		Main
    """
    training_set = pd.read_csv("letter-recognition-training.csv", skipinitialspace=True,
		                             skiprows=0, names=COLUMNS)
    test_set = pd.read_csv("letter-recognition-test.csv", skipinitialspace=True,
                           skiprows=0, names=COLUMNS)
    prediction_set = pd.read_csv("letter-recognition-eval.csv", skipinitialspace=True,
		                               skiprows=0, names=COLUMNS)

    sess = tf.InteractiveSession()
    merged = tf.summary.merge_all()

    feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

    classifier = tf.estimator.DNNClassifier(
      		hidden_units=HIDDEN_UNITS,
	      	feature_columns=feature_cols,
		 	    model_dir="/tmp/letter-recognition-dnn",
			     n_classes=26,
			     label_vocabulary=LABEL_VOCABULARY)
                 # activation_fn=tf.nn.tanh)

    train_writer = tf.summary.FileWriter("/tmp/letter-recognition-dnn/train", sess.graph)
    test_writer = tf.summary.FileWriter("/tmp/letter-recognition-dnn/test")
    tf.global_variables_initializer().run()

	# Train model
    classifier.train(input_fn=get_input_fn(training_set), steps=50000)
	# Test model
    accuracy = classifier.evaluate(
		      input_fn=get_input_fn(
			                       test_set,
                          num_epochs=1,
                          shuffle=False))["accuracy"]
    print("\nTest Accuracy: {0:f}\n".format(accuracy))
	# Predict model
    predictions = classifier.predict(
		      input_fn=get_input_fn(
                          prediction_set,
                          num_epochs=1,
                          shuffle=False))
    predicted_classes = [p["classes"] for p in predictions]
    print("New Samples, Class Predictions:    {}\n".format(predicted_classes))

    train_writer.close()
    test_writer.close()

if __name__ == "__main__":
    tf.app.run()
