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
- 2 hidden layers, with, for the both, 100 units
- 26 output neurons (number of classes)
- Optimizer algorithm : ***Adagrad***
- Activation function : ***RELU***
- Learning rate : *0.1*

```python
classifier = tf.estimator.DNNClassifier(
    	hidden_units=[100, 100],
	    feature_columns=feature_cols,
		 	model_dir="/tmp/letter-recognition",
			 n_classes=26,
			 label_vocabulary=LABEL_VOCABULARY)
```

The model is trained on the ***16000*** examples of the [letter-recognition-training.csv](https://github.com/jfourmond/TensorFlowLetterRecognition/blob/master/letter-recognition-training.csv) file, tested on the ***3900*** examples of the [letter-recognition-test.csv](https://github.com/jfourmond/TensorFlowLetterRecognition/blob/master/letter-recognition-test.csv) file, and finally evaluated on the 100 ***examples*** of the [letter-recognition-eval.csv](https://github.com/jfourmond/TensorFlowLetterRecognition/blob/master/letter-recognition-eval.csv) file.

The model is saved in the directory "*/tmp/letter-recognition*".

### Visualisation

TensorBoard can be used in order to visualize the graph, or the learning process, by specifying the model directory...

``` $> tensorboard --logdir=/tmp/letter-recognition```

### Accuracy

#### Machine Configuration

- OS : Windows 7 Professionnel 64 bits
- CPU : Intel(R) Core(TM)2 Duo CPU P8600 @ 2.40GHz 2.39GHz
- RAM : 4 Go

#### Accuracy with one layer

##### Activation Function : ReLU

| Layer 1 | Loss Final Step | Accuracy | Execution Time (s) |
|--------:|----------------:|---------:|-------------------:|
|      16 |         86.8474 |    0.775 |            395.045 |
|      26 |         100.841 |    0.792 |             340.08 |
|      32 |         94.1542 |    0.856 |            365.525 |
|      38 |         43.0134 |    0.878 |            367.517 |
|      42 |         39.6163 |     0.88 |            395.691 |
|      46 |          28.015 |    0.878 |            361.954 |
|      50 |         29.4245 |     0.89 |            415.047 |
|      60 |         26.7953 |     0.89 |            365.269 |
|      70 |         33.4799 |     0.91 |            344.694 |
|      80 |         35.0041 |      0.9 |            327.182 |
|      90 |         28.4253 |    0.917 |             373.81 |
|     100 |          28.844 |    0.917 |            396.768 |

##### Activation Function : TANH

| Layer 1 | Loss Final Step | Accuracy | Execution Time (s) |
|--------:|----------------:|---------:|-------------------:|
|      16 |         96.2142 |    0.776 |            371.515 |
|      26 |         82.3591 |    0.818 |            369.788 |
|      32 |         52.6221 |    0.860 |            365.063 |
|      38 |         56.6535 |    0.852 |            363.647 |
|      42 |         35.3007 |    0.875 |             414.47 |
|      46 |         42.2612 |    0.888 |            368.997 |
|      50 |         43.5843 |     0.89 |            362.113 |
|      60 |         31.3197 |     0.90 |            356.527 |
|      70 |         27.8964 |    0.922 |            351.236 |
|      80 |          25.739 |    0.926 |            344.032 |
|      90 |         18.6947 |    0.935 |            345.419 |
|     100 |         13.5394 |    0.942 |            349.571 |

#### Accuracy with two layers

##### Activation Function : ReLU

| Layer 1 | Layer 2 | Loss Final Step | Accuracy | Execution Time (s) |
|--------:|--------:|----------------:|---------:|-------------------:|
|      16 |      16 |         77.2852 |    0.834 |            397.938 |
|      26 |      26 |         56.6615 |    0.866 |            352.439 |
|      32 |      32 |         17.5571 |    0.901 |            366.805 |
|      38 |      38 |         29.7124 |    0.905 |            364.813 |
|      42 |      42 |         11.7265 |    0.931 |            420.828 |
|      46 |      46 |          19.728 |    0.916 |            376.747 |
|      50 |      50 |         15.7855 |     0.93 |            394.404 |
|      60 |      60 |         9.58298 |    0.942 |              353.1 |
|      70 |      70 |          4.3803 |    0.947 |            381.557 |
|      80 |      80 |         5.05916 |     0.95 |            436.446 |
|      90 |      90 |         1.87851 |    0.954 |            412.518 |
|     100 |     100 |         2.55939 |    0.953 |            407.731 |

##### Activation Function : TANH

| Layer 1 | Layer 2 | Loss Final Step | Accuracy | Execution Time (s) |
|--------:|--------:|----------------:|---------:|-------------------:|
|      16 |      16 |         60.5835 |    0.798 |            395.813 |
|      26 |      26 |         57.7649 |    0.872 |            358.864 |
|      32 |      32 |         19.4711 |    0.906 |            359.826 |
|      38 |      38 |         15.0797 |    0.925 |             419.92 |
|      42 |      42 |         15.7873 |    0.931 |            408.879 |
|      46 |      46 |          8.5714 |    0.937 |            391.825 |
|      50 |      50 |         10.6957 |     0.94 |            483.145 |
|      60 |      60 |         6.74168 |    0.944 |            344.709 |
|      70 |      70 |         3.99268 |    0.948 |            357.918 |
|      80 |      80 |         2.34741 |    0.958 |            399.283 |
|      90 |      90 |         1.79646 |    0.958 |            423.777 |
|     100 |     100 |          1.4419 |    0.964 |            401.549 |

#### Accuracy with three layers

##### Activation Function : ReLU

| Layer 1 | Layer 2 | Layer 3 | Loss Final Step | Accuracy | Execution Time (s) |
|--------:|--------:|--------:|----------------:|---------:|-------------------:|
|      16 |      16 |      16 |         54.2528 |    0.830 |            394.468 |
|      26 |      26 |      26 |          40.043 |    0.902 |            430.576 |
|      32 |      32 |      32 |         10.3218 |    0.914 |            365.758 |
|      38 |      38 |      38 |         2.70372 |    0.934 |            452.007 |
|      42 |      42 |      42 |         2.83129 |    0.937 |            430.211 |
|      46 |      46 |      46 |         6.77854 |    0.942 |            384.687 |
|      50 |      50 |      50 |         3.49199 |    0.941 |            428.058 |
|      60 |      60 |      60 |        0.471283 |    0.944 |            376.509 |
|      70 |      70 |      70 |        0.569922 |    0.955 |            371.872 |
|      80 |      80 |      80 |        0.409487 |    0.955 |            448.679 |
|      90 |      90 |      90 |         0.17778 |    0.952 |            472.883 |
|     100 |     100 |     100 |        0.528759 |    0.956 |            487.679 |

##### Activation Function : TANH

| Layer 1 | Layer 2 | Layer 3 | Loss Final Step | Accuracy | Execution Time (s) |
|--------:|--------:|--------:|----------------:|---------:|-------------------:|
|      16 |      16 |      16 |         57.3608 |    0.839 |            348.569 |
|      26 |      26 |      26 |         35.0019 |    0.894 |            383.372 |
|      32 |      32 |      32 |         19.8149 |    0.911 |            362.786 |
|      38 |      38 |      38 |         6.51512 |    0.934 |            371.123 |
|      42 |      42 |      42 |         3.98528 |    0.932 |            395.863 |
|      46 |      46 |      46 |         6.25795 |    0.940 |             394.53 |
|      50 |      50 |      50 |         3.87826 |    0.943 |            400.999 |
|      60 |      60 |      60 |         1.15276 |    0.945 |            362.097 |
|      70 |      70 |      70 |        0.363803 |    0.957 |            382.472 |
|      80 |      80 |      80 |        0.384293 |    0.957 |             454.97 |
|      90 |      90 |      90 |        0.498106 |    0.961 |            450.457 |
|     100 |     100 |     100 |        0.929079 |    0.961 |            463.845 |