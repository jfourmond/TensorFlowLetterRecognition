"""
    Deep Image Recognition
"""
import argparse
import itertools
import os
import time

from sklearn import preprocessing

import tensorflow as tf

IMAGE_SIZE = 20

PATH_TRAIN = "data/train/"
PATH_TEST = "data/test/"
PATH_PREDICT = "data/predict/"

LABELS_VOCABULARY = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
                     "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Logging activation
tf.logging.set_verbosity(tf.logging.INFO)

# Encodage des labels
LB = preprocessing.LabelEncoder()
LB.fit(LABELS_VOCABULARY)

Sess = tf.Session()

def distorted_parse(filename, label):
    """
        Lecture et déformation de l'image du fichier
    """
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    image_decoded = tf.cast(image_decoded, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(image_decoded, [height, width, 3])
    # TODO Remove : Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # TODO Randomly translate ?
    # TODO
    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

    return distorted_image, label

def parse(filename, label):
    """
        Lecture (sans déformation) de l'image du fichier
    """
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    image_decoded = tf.cast(image_decoded, tf.float32)

    return image_decoded, label

def fetch_dataset(data_dir, distorted=False):
    """
        Lecture des images du dataset
    """
    filenames = []
    labels = []

    for LABEL in LABELS_VOCABULARY:
        if os.access(data_dir + LABEL, os.F_OK):
            for elem in os.listdir(data_dir + LABEL):
                filenames.append(data_dir + LABEL + "/" + elem)
                labels.append(LABEL)

    size = len(filenames)

    filenames = tf.constant(filenames)

    labels = LB.transform(labels)
    labels = tf.constant(labels)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    if distorted:
        dataset = dataset.map(distorted_parse)
    else:
        dataset = dataset.map(parse)

    return dataset, size

def input_fn(dataset, n_steps, shuffle, batch_size):
    """Input function"""
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size)
    if  not n_steps is None and n_steps > 0:
        dataset = dataset.repeat(n_steps)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels

def cnn_model_fn(features, labels, mode):
    """Model function for CNN"""
    # Input Layer
    # Output : [batch_size, 20, 20, 3]
    input_layer = tf.reshape(features, [-1, 20, 20, 3])
    # Convolutional Layer #1
    # Input : [batch_size, 20, 20, 3]
    # Output : [batch_size, 20, 20, 36]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=36,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # Pooling Layer #1
    # Input : [batch_size, 20, 20, 36]
    # Output : [batch_size, 10, 10, 36]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # Convolutional Layer #2
    # Input : [batch_size, 10, 10, 36]
    # Output : [batch_size, 10, 10, 68]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=68,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # Pooling Layer #2
    # Input : [batch_size, 10, 10, 68]
    # Output : [batch_size, 5, 5, 68]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # Dense Layer
    # Input : [batch_size, 5, 5, 68]
    # Output : [batch_size, 1700]
    pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 68])
    # Input : [batch_size, 1700]
    # Output : [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    # Dropout method
    # Input : [batch_size, 1024]
    # Output : [batch_size, 1024]
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    # Logits Layer
    # Input : [batch_size, 1024]
    # Output : [batch_size, 26]
    logits = tf.layers.dense(inputs=dropout, units=26)
    # Generate predictions
    # Input : [batch_size, 26]
    predictions = {
        "classes" : tf.argmax(input=logits, axis=1),
        "probabilities" :  tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    # Calculate Loss
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=26)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    # Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    # Evaluation metrics (for EVAL mode)
    eval_metrics_ops = {
        "accuracy" : tf.metrics.accuracy(labels=labels, predictions=predictions["classes"]),
        "recall" : tf.metrics.recall(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)

def check_dataset(dataset):
    """Input function"""
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    i = 1
    while True:
        try:
            value = Sess.run(next_element)
            if value[0].shape[0] != 20:
                raise Exception("Wrong format", "{}, n°{}".format(value[1], i))
            if value[0].shape[1] != 20:
                raise Exception("Wrong format", "{}, n°{}".format(value[1], i))
            if value[0].shape[2] != 3:
                raise Exception("Wrong format", "{}, n°{}".format(value[1], i))
            i = i+1
        except tf.errors.OutOfRangeError:
            break

def main(_):
    """
		Main
    """
    N_STEPS = FLAGS.n_steps
    BATCH_SIZE = FLAGS.batch_size
    MODEL_DIR = FLAGS.model_dir
    SHOW_PREDICT = FLAGS.show_predict

    print("NOMBRE DE PAS : {}".format(N_STEPS))
    print("TAILLE DES BATCHS : {}".format(BATCH_SIZE))
    print("REPERTOIRE DU MODELE : {}".format(MODEL_DIR))
    print("AFFICHAGE DES PREDICTIONS : {}".format(SHOW_PREDICT))

    # Lecture des images pour l'apprentissage
    print("Lecture des images pour l'apprentissage...")
    train_dataset, train_dataset_size = fetch_dataset(PATH_TRAIN, True)
    check_dataset(train_dataset)
    print("FAIT : {} IMAGES".format(train_dataset_size))

    # Lecture des images pour l'évaluation
    print("Lecture des images pour l'évaluation...")
    test_dataset, test_dataset_size = fetch_dataset(PATH_TEST)
    check_dataset(test_dataset)
    print("FAIT : {} IMAGES".format(test_dataset_size))

    # Lecture des images pour la prédiction
    if FLAGS.show_predict:
        print("Lecture des images pour la prédiction...")
        predict_dataset, predict_dataset_size = fetch_dataset(PATH_PREDICT)
        check_dataset(predict_dataset)
        print("FAIT : {} IMAGES".format(predict_dataset_size))

    # Construction du réseau de neurones convolutifs
    print("Construction du réseau de neurones convolutifs...")
    cnn = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=MODEL_DIR)
    print("FAIT")

    tensors_to_log = {"probabilities" : "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    start = time.time()

    # Entraînement du réseau de neurones convolutifs
    print("Entraînement du réseau de neurones convolutifs...")
    cnn.train(input_fn=lambda: input_fn(train_dataset, N_STEPS, True, BATCH_SIZE),
              hooks=[logging_hook])
    print("FAIT")

    # Evaluation du réseau de neurones convolutifs
    print("Evaluation du réseau de neurones convolutifs...")
    eval_results = cnn.evaluate(
        input_fn=lambda: input_fn(test_dataset, None, False, BATCH_SIZE))
    print("FAIT : {}".format(eval_results))

    # Prédiction du réseau de neurones convolutifs
    if SHOW_PREDICT:
        y = cnn.predict(input_fn=lambda: input_fn(predict_dataset, None, False, BATCH_SIZE))
        pred = list(p["classes"] for p in y)
        needed_labels = []
        # Récupération des labels voulus (pour comparaison)
        iterator = predict_dataset.make_one_shot_iterator()
        _, labels = iterator.get_next()
        sess = tf.Session()
        while True:
            try:
                needed_labels.append(sess.run(labels))
            except tf.errors.OutOfRangeError:
                break
        sess.close()
        print("Résultats voulus : {}".format(str(LB.inverse_transform(needed_labels))))
        print("Predictions: {}".format(str(LB.inverse_transform(pred))))

    end = time.time() - start
    minutes, seconds = divmod(end, 60)
    print("Temps d'exécution : {}m {:.4f}s".format(int(minutes), seconds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_steps',
        '-n',
        type=int,
        default=100,
        help='Number of steps for which to train model (default 100)'
    )
    parser.add_argument(
        '--batch_size',
        '-b',
        type=int,
        default=100,
        help='Size of the batches (default 100)'
    )
    parser.add_argument(
        '--model_dir',
        '-md',
        type=str,
        default='/tmp/letter',
        help='Directory where the checkpoint and the model will be stored (default \'/tmp/letter\')'
    )
    parser.add_argument(
        '--show_predict',
        '-p',
        type=bool,
        default=False,
        help='Show the prediction'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run()
    