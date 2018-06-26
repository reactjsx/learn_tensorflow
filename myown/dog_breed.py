import tensorflow as tf
import os
import numpy as np
from PIL import Image
import pandas as pd
import collections
#import matplot.pyplot as plt

NUM_CLASSES = 120
BATCH_SIZE = 32
INITIAL_LEARNING_RATE = 0.001
NUM_EPOCHS_PER_DECAY = 350
LEARNING_RATE_DECAY_FACTOR = 0.1

def _create_variable(name, shape, initializer, dtype):
  var = tf.get_variable(
    name, shape, initializer=initializer, dtype=dtype)
  return var

def _get_weights(name, shape, stddev):
  var = _create_variable(
    name,
    shape,
    tf.truncated_normal_initializer(stddev=stddev),
    dtype=tf.float32)
  return var

def _get_biases(name, shape, init_value=0.):
  var = _create_variable(
    name, shape,
    tf.constant_initializer(init_value),
    dtype=tf.float32)
  return var

def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [224, 224]) / 255.
  image_resized.set_shape([224, 224, 3])
  label = tf.cast(label, tf.int64)
  return image_resized, label

def _read_from_file(filepath, folder):
  train_data = pd.read_csv(filepath_or_buffer=filepath,
                           names=['id', 'breed'],
                           header=0)
  train_data = train_data.values
  filenames = train_data[:, 0]
  filenames = [os.path.join(folder, f + '.jpg') for f in filenames]
  labels = train_data[:, 1]
  counter = collections.Counter(labels)
  all_breed_list, _ = list(zip(*counter.most_common()))
  all_breed = dict(zip(all_breed_list, range(len(all_breed_list))))
  int_labels = [all_breed[lbl] for lbl in labels]
  return all_breed_list, all_breed, filenames, int_labels

def _input_fn(filenames, int_labels):
  dataset = tf.data.Dataset.from_tensor_slices((filenames, int_labels))
  dataset = dataset.map(_parse_function)

  iterator = dataset.shuffle(20000).repeat().batch(BATCH_SIZE).make_one_shot_iterator()
  next_element = iterator.get_next()
  return next_element

def _conv2d(name, input_, out_channel):
  with tf.variable_scope(name) as scope:
    weights = _get_weights('weights',
                           shape=[3, 3, input_.get_shape()[-1], out_channel],
                           stddev=5e-2)
    biases = _get_biases('biases', [out_channel], init_value=0.)
    tf.summary.histogram(name + '_weights', weights)
    conv = tf.nn.conv2d(input_, weights,
                        [1, 1, 1, 1],
                        padding='SAME')
    conv = tf.nn.bias_add(conv, biases)
    conv = tf.nn.relu(conv, name=scope.name)
    tf.summary.histogram(name, conv)
  return conv

def _pooling2d(name, conv):
  with tf.variable_scope(name) as scope:
    pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1],
                         padding='SAME', name='pool')
  return pool

def _dense_and_activate(name, input_, out_channel,
                        need_flatten=False, activate=tf.nn.relu):
  with tf.variable_scope(name) as scope:
    dense_shape = input_.get_shape()
    if need_flatten:
      input_ = tf.reshape(input_, [BATCH_SIZE, -1])
      in_channel = dense_shape[1] * dense_shape[2] * dense_shape[3]
    else:
      in_channel = dense_shape[1]
    weights = _get_weights(
      'weights',
      shape=[in_channel, out_channel],
      stddev=0.04)
    tf.summary.histogram(name + '_weights', weights)
    biases = _get_biases('biases', [out_channel],
                         init_value=0.1)
    if activate:
      fc = activate(tf.matmul(input_, weights) + biases, name=scope.name)
    else:
      fc = tf.add(tf.matmul(input_, weights),
                  biases,
                  name=scope.name)
    tf.summary.histogram(name, fc)
    return fc

def net(image):
  conv1_1 = _conv2d('conv1_1', image, 64)
  conv1_2 = _conv2d('conv1_2', conv1_1, 64)
  pool1 = _pooling2d('pool1', conv1_2)

  conv2_1 = _conv2d('conv2_1', pool1, 128)
  conv2_2 = _conv2d('conv2_2', conv2_1, 128)
  pool2 = _pooling2d('pool2', conv2_2)

  conv3_1 = _conv2d('conv3_1', pool2, 128)
  conv3_2 = _conv2d('conv3_2', conv3_1, 128)
  conv3_3 = _conv2d('conv3_3', conv3_2, 128)
  pool3 = _pooling2d('pool3', conv3_3)

  conv4_1 = _conv2d('conv4_1', pool3, 256)
  conv4_2 = _conv2d('conv4_2', conv4_1, 256)
  conv4_3 = _conv2d('conv4_3', conv4_2, 256)
  pool4 = _pooling2d('pool4', conv4_3)

  conv5_1 = _conv2d('conv5_1', pool4, 512)
  conv5_2 = _conv2d('conv5_2', conv5_1, 512)
  conv5_3 = _conv2d('conv5_3', conv5_2, 512)
  pool5 = _pooling2d('pool5', conv5_3)

  fc6 = _dense_and_activate('fc6', pool5, 1024, need_flatten=True)
  fc7 = _dense_and_activate('fc7', fc6, 512)
  logits = _dense_and_activate('softmax_linear', fc7, NUM_CLASSES, activate=None)
  return logits

def main(_):
  all_breed_list, all_breed, filenames, int_labels = _read_from_file('labels.csv', 'train')
  image, label = _input_fn(filenames, int_labels)

  logits = net(image)

  global_step = tf.train.get_or_create_global_step()
  decay_steps = int(10222 / BATCH_SIZE * NUM_EPOCHS_PER_DECAY)

  cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=label, logits=logits, name='cross_entropy_loss')
  loss = tf.reduce_mean(cross_entropy_loss, name='loss')
  tf.summary.scalar('Training Loss', loss)

  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('Learning Rate', lr)
  merged = tf.summary.merge_all()
  opt = tf.train.GradientDescentOptimizer(lr)
  grads = opt.compute_gradients(loss)
  train_op = opt.apply_gradients(
    grads, global_step)

  saver = tf.train.Saver()

  sess = tf.Session()
  train_writer = tf.summary.FileWriter('model/train', sess.graph)
  sess.run(tf.global_variables_initializer())
  for i in range(20000):
    _ = sess.run(train_op)
    summary = sess.run(merged)
    train_writer.add_summary(summary, i)
    if i % 100 == 0:
      print('Iteration: {}'.format(i * BATCH_SIZE))
      loss_value, logit_values, label_values = sess.run([loss, logits, label])
      print('Loss: {}'.format(loss_value / BATCH_SIZE))
      logit_values = np.argmax(logit_values, axis=1)
      label_names = [all_breed_list[i] for i in label_values]
      logit_names = [all_breed_list[i] for i in logit_values]
      print('Labels: {}'.format(label_values))
      print('Logits: {}'.format(logit_values))
      print('Predicted Breed: {}'.format(logit_names))
      saver.save(sess, 'model/model.ckpt')

if __name__ == '__main__':
  tf.app.run()