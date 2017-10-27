# Letter Recognition

Python Deep Learning on the "Letter Image Recognition Data" with TensorFlow

---

## Data

The "[Letter Image Recognition Data](https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/)" comes from the [the UC Irvine Machine Learning Repository](http://archive.ics.uci.edu/ml/).

As specifiy in the [data description](https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.names), the dataset is composed of 17 attributes (one label and 16 features).

Its main goal is :
> The objective is to identify each of a large number of black-and-white rectangular pixel displays as one of the 26 capital letters in the English alphabet.  The character images were based on 20 different fonts and each letter within these 20 fonts was randomly distorted to produce a file of 20,000 unique stimuli.  Each stimulus was converted into 16 primitive numerical attributes (statistical moments and edge counts) which were then scaled to fit into a range of integer values from 0 through 15.  We typically train on the first 16000 items and then use the resulting model to predict the letter category for the remaining 4000.

This isn't image recognition.

This project was made to learn TensorFlow, and use one Deep Neural Network for classification, similarly as the Housing Data from the [TensorFlow library Get Started Page](https://www.tensorflow.org/get_started/input_fn).

The original dataset has been divided in 3 parts : the training dataset, the test dataset, and the evaluation dataset.

## Python Script

```$> python letter-recognition.py```

### Model

The current model of the python script is a Deep Neural Network Classifier with the following caracteristics :
- 2 hidden layers, with, for the both, 40 units
- 26 output neurons (number of classes)
- Optimizer algorithm : ***Adagrad***
- Activation function : ***RELU***
- Learning rate : *0.1*

```python
classifier = tf.estimator.DNNClassifier(
    	hidden_units=[40, 40],
	    feature_columns=feature_cols,
		 	model_dir="/tmp/letter-recognition",
			 n_classes=26,
			 label_vocabulary=LABEL_VOCABULARY)
```

The model is saved in the repertory "*/tmp/letter-recognition*".

### Visualisation

TensorBoard can be used in order to visualize the graph, or the learning process...

``` $> tensorboard --logdir=/tmp/letter-recognition```

### Accuracy

| Layer 1 | Layer 2 | Layer 3 | Activation Function | Loss Final Step | Accuracy | Execution Time |
|---------|---------|---------|---------------------|-----------------|----------|----------------|
|      16 |         |         | RELU                |         86.8474 |    0.775 |        395.045 |
|         |         |         | TANH                |         96.2142 |    0.776 |        371.515 |
|      16 |      16 |         | RELU                |         77.2852 |    0.834 |        397.938 |
|      16 |      16 |         | TANH                |                 |          |                |
