import tensorflow as tf
from six.moves import urllib
import os

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data'
urllib.request.urlretrieve(url, 'data')

with open('data', 'r') as f:
  a = f.readlines()
  data = a[1:]
  data = [x.strip('\n').replace(' ', '').split(',') for x in data]

def create_data():
  features = []
  labels = []

  for element in data:
    features.append([float(e) for e in element[:-1]])
    labels.append(float(element[-1]))

  features = np.array(features)
  feature_names = ['recency', 'frequency', 'monetary', 'time']
  new_features = {}

  for i in range(features.shape[1]):
    max_value = np.max(features[:, i])
    features[:, i] = (features[:, i] - max_value / 2) / max_value * 2
    new_features[feature_names[i]] = features[:, i]
  
  return new_features, labels

def _input_fn(features, labels, batch_size):
  dataset = tf.data.Dataset.from_tensor_slices((features, labels))
  dataset = dataset.shuffle(800).repeat().batch(batch_size)
  
  iterator = dataset.make_one_shot_iterator()
  
  return iterator.get_next()

recency_feature = tf.feature_column.numeric_column(key='recency')
frequency_feature = tf.feature_column.numeric_column(key='frequency')
monetary_feature = tf.feature_column.numeric_column(key='monetary')
time = tf.feature_column.numeric_column(key='time')

my_feature_columns = [recency_feature, frequency_feature, monetary_feature, time]

classifier = tf.estimator.DNNClassifier(
  feature_columns=my_feature_columns,
  hidden_units=[10, 10])

features, labels = create_data()
classifier.train(
  input_fn=lambda: _input_fn(features, labels, 100),
  steps=10000)
