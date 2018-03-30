"""
    Deep Image Recognition
"""
import argparse
import itertools
import os
import time

from sklearn import preprocessing

import tensorflow as tf

PATH_TRAIN = "data/train/"
PATH_TEST = "data/test/"
PATH_PREDICT = "data/predict/"

LABELS_VOCABULARY = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
                     "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

FLAGS = None

# Logging activation
tf.logging.set_verbosity(tf.logging.INFO)

# Encodage des labels
LB = preprocessing.LabelEncoder()
LABELS_FITTED = LB.fit(LABELS_VOCABULARY)

Sess = tf.Session()

def parse(filename, label):
    """
        Lecture de l'image du fichier
    """
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    image_decoded = tf.cast(image_decoded, tf.float32)

    return image_decoded, label

def fetch_train_dataset():
    """ Création du jeu de données d'apprentissage"""
    filenames = []
    labels = []

    for LABEL in LABELS_VOCABULARY:
        if os.access(PATH_TRAIN + LABEL, os.F_OK):
            for elem in os.listdir(PATH_TRAIN + LABEL):
                filenames.append(PATH_TRAIN + LABEL + "/" + elem)
                labels.append(LABEL)

    size = len(filenames)

    filenames = tf.constant(filenames)

    labels = LB.transform(labels)
    labels = tf.constant(labels)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse)

    return dataset, size

def fetch_test_dataset():
    """ Création du jeu de données d'évaluation"""
    filenames = []
    labels = []

    for LABEL in LABELS_VOCABULARY:
        if os.access(PATH_TEST + LABEL, os.F_OK):
            for elem in os.listdir(PATH_TEST + LABEL):
                filenames.append(PATH_TEST + LABEL + "/" + elem)
                labels.append(LABEL)

    size = len(filenames)

    filenames = tf.constant(filenames)

    labels = LB.transform(labels)
    labels = tf.constant(labels)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse)

    return dataset, size

def fetch_predict_dataset():
    """ Création du jeu de données de prédiction"""
    filenames = []
    labels = []

    for LABEL in LABELS_VOCABULARY:
        if os.access(PATH_PREDICT + LABEL, os.F_OK):
            for elem in os.listdir(PATH_PREDICT + LABEL):
                filenames.append(PATH_PREDICT + LABEL + "/" + elem)
                labels.append(LABEL)

    size = len(filenames)

    filenames = tf.constant(filenames)

    labels = LB.transform(labels)
    labels = tf.constant(labels)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse)

    return dataset, size

def input_fn(dataset, n_steps, shuffle, batch_size):
    """Input function"""
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
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
            if value[0].shape[2] != 3:
                raise Exception("Wrong format", "{}, n°{}".format(value[1], i))
            i = i+1
        except tf.errors.OutOfRangeError:
            break


feature_spec = {'image/encoded': tf.FixedLenFeature(shape=[], dtype=tf.string) }

def serving_input_receiver_fn():
    """An input receiver that expects a serialized tf.Example."""
    serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[1], name='input_image')
    receiver_tensors = {'examples': serialized_tf_example}
    features = tf.parse_example(serialized_tf_example, feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

def main(_):
    """
		Main
    """
    N_STEPS = FLAGS.n_steps
    MODEL_DIR = FLAGS.model_dir
    SHOW_PREDICT = FLAGS.show_predict
    EXPORT_DIR = FLAGS.export_dir

    print("NOMBRE DE PAS : {}".format(N_STEPS))
    print("REPERTOIRE DU MODELE : {}".format(MODEL_DIR))
    print("AFFICHAGE DES PREDICTIONS : {}".format(SHOW_PREDICT))
    print("REPERTOIRE D'EXPORTATION : {}".format(EXPORT_DIR))

    # Lecture des images pour l'apprentissage
    print("Lecture des images pour l'apprentissage...")
    train_dataset, train_dataset_size = fetch_train_dataset()
    check_dataset(train_dataset)
    print("FAIT : {} IMAGES".format(train_dataset_size))

    # Lecture des images pour l'évaluation
    print("Lecture des images pour l'évaluation...")
    test_dataset, test_dataset_size = fetch_test_dataset()
    check_dataset(test_dataset)
    print("FAIT : {} IMAGES".format(test_dataset_size))

    # Lecture des images pour la prédiction
    if FLAGS.show_predict:
        print("Lecture des images pour la prédiction...")
        predict_dataset, predict_dataset_size = fetch_predict_dataset()
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
    cnn.train(input_fn=lambda: input_fn(train_dataset, N_STEPS, True, train_dataset_size),
              hooks=[logging_hook])
    print("FAIT")

    # Evaluation du réseau de neurones convolutifs
    print("Evaluation du réseau de neurones convolutifs...")
    eval_results = cnn.evaluate(
        input_fn=lambda: input_fn(test_dataset, None, False, test_dataset_size))
    print("FAIT : {}".format(eval_results))

    # Prédiction du réseau de neurones convolutifs
    if SHOW_PREDICT:
        y = cnn.predict(input_fn=lambda: input_fn(predict_dataset, None, False, predict_dataset_size))
        pred = list(p["classes"] for p in y)
        print("Predictions: {}".format(str(LB.inverse_transform(pred))))

    if EXPORT_DIR:
        # path = cnn.export_savedmodel(EXPORT_DIR, serving_input_receiver_fn)

        image = tf.placeholder(tf.string, [None, 20, 20, 3])
        serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'image': image,
        })
        path = cnn.export_savedmodel(EXPORT_DIR, serving_input_fn)
        print("Exportation dans le répertoire : {}".format(path))

    end = time.time() - start
    minutes, seconds = divmod(end, 60)
    print("Temps d'exécution : {}m {:.4f}s".format(int(minutes), seconds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_steps',
        '-n',
        type=int,
        default=1000,
        help='Number of steps for which to train model'
    )
    parser.add_argument(
        '--model_dir',
        '-md',
        type=str,
        default='/tmp/cnn',
        help='Directory where the checkpoint and the model will be stored'
    )
    parser.add_argument(
        '--show_predict',
        '-p',
        type=bool,
        default=False,
        help='Show the prediction'
    )
    parser.add_argument(
        '--export_dir',
        '-e',
        type=str,
        default=None,
        help='Directory where the model will be exported'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run()
    