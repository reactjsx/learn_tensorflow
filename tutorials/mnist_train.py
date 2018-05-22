import numpy
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def load_data():
    return tf.keras.datasets.mnist.load_data()

def _input_fn(features, labels, batch_size):
    features = (features - 127) / 127.
    features = {'x': features}
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(70000).repeat().batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

def cnn_model_fn(features, labels, mode):
    # Input layer: shape [batch_size, image_height, image_width, channels]
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])

    # Convolution Layer 1
    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5], # [height, width], can be specified as kernel_size=5
            padding='same',
            activation=tf.nn.relu)

    # Pooling Layer 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolution Layer 2
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu)

    # Pooling Layer 2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
            'classes': tf.argmax(input=logits, axis=1),
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculation Loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure training op
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics
    eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(
                labels=labels, predictions=predictions['classes'])}
    return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(argv):
    # Load data
    (train_data, train_labels), (eval_data, eval_labels) = load_data()

    # Logging hook
    tensors_to_log = {'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

    mnist_classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn,
            model_dir='model/mnist')
    mnist_classifier.train(
            input_fn=lambda: _input_fn(train_data, train_labels, 100),
            steps=20000,
            hooks=[logging_hook])

    # Evaluate the model
    eval_results = mnist_classifier.evaluate(
            input_fn=lambda: _input_fn(eval_data, eval_labels, None))
    print(eval_results)


if __name__ == '__main__':
    tf.app.run()
