# -*- coding: utf-8 -*-
"""Hello, Colaboratory

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/notebooks/welcome.ipynb
"""

import tensorflow as tf

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))

dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]),
                                             tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3)

dataset = tf.data.Dataset.from_tensor_slices(
    {'a': tf.random_uniform([4]),
     'b': tf.random_uniform([4, 10], maxval=100, dtype=tf.int32)})
print(dataset.output_types)
print(dataset.output_shapes)

dataset = tf.data.Dataset.range(1000)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

sess = tf.Session()
for i in range(1000):
  value = sess.run(next_element)
  assert i == value

max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

sess.run(iterator.initializer, feed_dict={max_value: 10})
for i in range(10):
  value = sess.run(next_element)
  assert i == value

sess.run(iterator.initializer, feed_dict={max_value: 100})
for i in range(100):
  value = sess.run(next_element)
  assert i == value

training_dataset = tf.data.Dataset.range(100).map(
  lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.data.Dataset.range(50)

iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

for _ in range(20):
  sess.run(training_init_op)
  for _  in range(100):
    sess.run(next_element)
   
  sess.run(validation_init_op)
  for _ in range(50):
    sess.run(next_element)

sess = tf.Session()
training_dataset = tf.data.Dataset.range(100).map(
  lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
validation_dataset = tf.data.Dataset.range(50)

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
  handle, training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()

training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

training_handle = sess.run(training_iterator.string_handle())
validation_handle = sess.run(validation_iterator.string_handle())

i = 0
while i < 1000:
  for _ in range(200):
    sess.run(next_element, feed_dict={handle: training_handle})
  
  sess.run(validation_iterator.initializer)
  for _ in range(50):
    sess.run(next_element, feed_dict={handle: validation_handle})
  i += 1

dataset = tf.data.Dataset.range(5) #.repeat()
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

result = tf.add(next_element, next_element)

sess.run(iterator.initializer)
print(sess.run(result))
print(sess.run(result))
print(sess.run(result))
print(sess.run(result))
print(sess.run(result))
print(sess.run(result))
try:
  sess.run(result)
except tf.errors.OutOfRangeError:
  print('End')

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 100])))
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

iterator = dataset3.make_initializable_iterator()

sess.run(iterator.initializer)
next1, (next2, next3) = iterator.get_next()

import numpy as np

# with np.load('/var/data/training_data.npy') as data:
#   features = data['features']
#   labels = data['labels']

features = np.random.uniform(-5, 5, (10, 4))
labels = np.random.uniform(-5, 5, (10,))

assert features.shape[0] == labels.shape[0]

# BAD FOR BUFFER MEMORY
# dataset = tf.data.Dataset.from_tensor_slices((features, labels))

# Instead use tf.placeholder
features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
iterator = dataset.make_initializable_iterator()

sess = tf.Session()
sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                          labels_placeholder: labels})

# filenames = ['/var/data/file1.tfrecord', '/var/data/file2.tfrecord']

filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(lambda x: x) # process record to tensors
dataset = dataset.repeat()
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()

training_filenames = ['/var/data/file1.tfrecord', '/var/data/file2.tfrecord']
sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

validation_filenames = ['/var/data/validation.tfrecord']
sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})

filenames = ['/var/data/file1.txt', '/var/data/file2.txt']
dataset = tf.data.Dataset.from_tensor_slices(filenames)

dataset = dataset.flat_map(
  lambda filename: (
    tf.data.TextLineDataset(filename)
    .skip(1)
    .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), '#'))))

def _parse_function(example_proto):
  features = {'image': tf.FixedLenFeature((), tf.string, default_value=''),
              'label': tf.FixedLenFeature((), tf.int32, default_value=0)}
  parsed_features = tf.parse_single_example(example_proto, features)
  return parsed_features['image'], parsed_features['label']

filenames = ['/var/data/file1.tfrecord', '/var/data/file2.tfrecord']
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)

def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string)
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

filenames = tf.constant(['/var/data/image1.jpg', '/var/data/image2.jpg'])
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)

import cv2

def _read_py_function(filename, label):
  image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
  return image_decoded, label

def _resize_function(image_decoded, label):
  image_decoded.set_shape([None, None, None])
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

filenames = ['/var/data/image1.jpg', '/var/data/image2.jpg']
labels = [0, 20]

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(
  lambda filename, label: tuple(tf.py_func(
    _read_py_function, [filename, label], [tf.uint8, label.dtype])))
dataset = dataset.map(_resize_function)
