import tensorflow as tf

from sklearn import preprocessing

import argparse
import cv2
import warnings

FLAGS = None

LABELS_VOCABULARY = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
                     "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "empty"]

# Encodage des labels
LB = preprocessing.LabelEncoder()
LABELS_FITTED = LB.fit(LABELS_VOCABULARY)


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
    # Output : [batch_size, 27]
    logits = tf.layers.dense(inputs=dropout, units=27)
    # Generate predictions
    ### NEW PART
    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # Calculate Loss
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=27)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)
    
    # Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

def input_fn(dataset):
    """Input function"""
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

def predict(cnn, section):
    if(section.size > 0):
        section = tf.cast(section, tf.float32)
        section = tf.image.resize_images(section, [20, 20])
        dataset = tf.data.Dataset.from_tensor_slices(([section], [0]))
        preds = cnn.predict(input_fn=lambda: input_fn(dataset))
        # pred = list(p["classes"] for p in y)
        return preds

def change_selection(selection, change):
    if change == "up":
        return selection + 1
    elif change == "down":
        return selection - 1

def draw_section(original_img, top_left, bottom_right):
    return cv2.rectangle(original_img, top_left, bottom_right, (0, 255, 0), 1)


def get_section(original_img, top_left, bottom_right):
    return original_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

def main():
    PATH_MODEL = FLAGS.model_dir
    PATH_IMAGE = FLAGS.path_to_image

    SELECTION = 20

    readed = ''

    # Construction du réseau de neurones convolutifs
    print("Construction du réseau de neurones convolutifs...")
    cnn = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=PATH_MODEL)
    print("FAIT")

    top_left = (0, 0)
    bottom_right = (top_left[0] + SELECTION, top_left[1] + SELECTION)
    # Lecture de l'image
    original_img = cv2.imread(PATH_IMAGE)
    shape_image = original_img.shape    # Taille de l'image (sans la dimension RBG)
    print("IMAGE : {}".format(PATH_IMAGE))
    print("DIMENSION DE L'IMAGE : {}".format(shape_image))

    # Récupération de la section
    section = get_section(original_img, top_left, bottom_right)
    # Lecture d'une ligne
    while(top_left[0] < shape_image[1]):
        # Prédiction
        section = get_section(original_img, top_left, bottom_right)
        preds = predict(cnn, section)
        zip_preds = zip(preds)

        for zip_pred in zip_preds:
            pred = zip_pred[0]

            class_id = pred['class_ids'][0]
            probability = pred['probabilities'][class_id]

            label = LB.inverse_transform(class_id)

            print('\nPrediction is "{}" ({:.1f}%)'.format(label, probability * 100))

            if label == 'empty':
                readed = readed + ' '
            else:
                readed = readed + label

        top_left = (top_left[0] + SELECTION, top_left[1])
        bottom_right = (top_left[0] + SELECTION, top_left[1] + SELECTION)

    print('\nResults : {}' .format(readed))

if __name__ == "__main__":
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_dir',
        type=str,
        help='Path to the model'
    )
    parser.add_argument(
        'path_to_image',
        type=str,
        help='Path to the image which will be loaded'
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()