import collections
import math

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
import argparse

train_url = 'https://storage.googleapis.com/mledu-datasets/sparse-data-embedding/train.tfrecord'
train_path = tf.keras.utils.get_file(train_url.split('/')[-1], train_url)
test_url = 'https://storage.googleapis.com/mledu-datasets/sparse-data-embedding/test.tfrecord'
test_path = tf.keras.utils.get_file(test_url.split('/')[-1], test_url)

parser = argparse.ArgumentParser()
parser.add_argument('--clf', default='linear', help='linear or dnn or embedding')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def _parse_function(record):
    """Extracts features and labels from tfrecord

    Args:
        record: File path to a TFRecord file
    Returns:
        A (labels, features) tuple:
            features: A dict of tensors representing the features
            labels: A tensor with the corresponding labels
    """
    features = {
        'terms': tf.VarLenFeature(dtype=tf.string),
        'labels': tf.FixedLenFeature(shape=[1], dtype=tf.float32)
    }

    parsed_features = tf.parse_single_example(record, features)

    terms = parsed_features['terms'].values
    labels = parsed_features['labels']

    return {'terms': terms}, labels

def _input_fn(input_filenames, num_epochs=None, shuffle=True):
    """
    Function to parse the tf.Examples from files
    and split them into features and labels

    Args:
        input_filenames: TFRecord filepath
        num_epochs: number of training epochs
        shuffle: whether to shufle dataset
    Return:
        features: A dict of tensors representing the features
        labels: A tensor with corresponding labels
    """
    ds = tf.data.TFRecordDataset(input_filenames)
    ds = ds.map(_parse_function)
    if shuffle:
        ds = ds.shuffle(10000)

    ds = ds.padded_batch(25, ds.output_shapes)
    ds = ds.repeat(num_epochs)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def main(argv):
    args = parser.parse_args(argv[1:])

    informative_terms = ("bad", "great", "best", "worst", "fun", "beautiful",
                         "excellent", "poor", "boring", "awful", "terrible",
                         "definitely", "perfect", "liked", "worse", "waste",
                         "entertaining", "loved", "unfortunately", "amazing",
                         "enjoyed", "favorite", "horrible", "brilliant", "highly",
                         "simple", "annoying", "today", "hilarious", "enjoyable",
                         "dull", "fantastic", "poorly", "fails", "disappointing",
                         "disappointment", "not", "him", "her", "good", "time",
                         "?", ".", "!", "movie", "film", "action", "comedy",
                         "drama", "family")

    # Convert terms feature into categorical column
    terms_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(key='terms', vocabulary_list=informative_terms)

    my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    if args.clf == 'linear':
        feature_columns = [terms_feature_column]
        classifier = tf.estimator.LinearClassifier(
            feature_columns=feature_columns,
            optimizer=my_optimizer)

        classifier.train(
            input_fn=lambda: _input_fn([train_path]),
            steps=args.train_steps)

        evaluation_metrics = classifier.evaluate(
            input_fn=lambda: _input_fn([train_path]),
            steps=args.train_steps)
        print('Training set metrics:')
        for m in evaluation_metrics:
            print(m, evaluation_metrics[m])
        print('------------------')

        evaluation_metrics = classifier.evaluate(
            input_fn=lambda: _input_fn([test_path]),
            steps=args.train_steps)

        print('Test set metrics:')
        for m in evaluation_metrics:
            print(m, evaluation_metrics[m])
        print('------------------')
    elif args.clf == 'dnn':
        feature_columns = [tf.feature_column.indicator_column(terms_feature_column)]
        classifier = tf.estimator.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=[20, 20],
            optimizer=my_optimizer)

        try:
            classifier.train(
                input_fn=lambda: _input_fn([train_path]),
                steps=args.train_steps)
            evaluation_metrics = classifier.evaluate(
                input_fn=lambda: _input_fn([train_path]),
                steps=1)
            print('Training set metrics:')
            for m in evaluation_metrics:
                print(m, evaluation_metrics[m])
            print('------------------')

            evaluation_metrics = classifier.evaluate(
                input_fn=lambda: _input_fn([test_path]),
                steps=1)
            print('Test set metrics:')
            for m in evaluation_metrics:
                print(m, evaluation_metrics[m])
            print('------------------')
        except ValueError as err:
            print(err)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
