"""Implementation of a neural network for mnist from scratch"""


import math
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist


class NaiveDense():
  """NaiveDense"""

  def __init__(self, input_size, output_size, activation):
    self.activation = activation

    w_shape = (input_size, output_size)
    w_initial_value = tf.random.uniform(w_shape, 0.0, 1e-1)
    self.weights = tf.Variable(w_initial_value)

    b_shape = (output_size,)
    b_initial_value = tf.zeros(b_shape)
    self.bias = tf.Variable(b_initial_value)

  def __call__(self, inputs):
    return self.activation(tf.matmul(inputs, self.weights) + self.bias)

  @property
  def weights(self):
    """Returns weights and bias as [weights, bias]"""
    return [self.weights, self.bias]


class NaiveSequential():
  """NaiveSequential"""

  def __init__(self, layers):
    self.layers = layers

  def __call__(self, inputs):
    outputs = inputs
    for layer in self.layers:
      outputs = layer(outputs)
    return outputs

  @property
  def weights(self):
    """Returns weights from all layers"""
    weights = []
    for layer in self.layers:
      weights += layer.weights
    return weights


class BatchGenerator():
  """BatchGenerator"""

  def __init__(self, images, labels, batch_size=128):
    assert(len(images) == len(labels))
    self.index = 0
    self.images = images
    self.labels = labels
    self.batch_size = batch_size
    self.num_batches = math.ceil(len(images) / batch_size)

  def next(self):
    """Returns images,labels"""
    images = self.images[self.index: self.index + self.batch_size]
    labels = self.labels[self.index: self.index + self.batch_size]
    self.index += self.batch_size
    return images, labels

# move the weights slightly toward a closer solution


def update_weights(gradients, weights, learning_rate=1e-3):
  for gradient, weight in zip(gradients, weights):
    weight.assign_sub(gradient * learning_rate)  # -= in tensorflow


def one_training_step(model, images_batch, labels_batch):
  with tf.GradientTape() as tape:
    predictions = model(images_batch)
    per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(
        labels_batch, predictions)
    average_loss = tf.reduce_mean(per_sample_losses)
  gradients = tape.gradient(average_loss, model.weights)
  update_weights(gradients, model.weights)
  return average_loss


def test(model, images, labels):
  predictions = model(images)
  predictions = predictions.numpy()
  predicted_labels = np.argmax(predictions, axis=1)
  matches = predicted_labels == test_labels
  accuracy = matches.mean()
  print(f"Test accuracy on {len(labels)} labels = {accuracy:.2f}")


def fit(model, images, labels, batch_size=128, test_fn=None):
  for epoch_counter in range(10):
    batch_generator = BatchGenerator(images, labels, batch_size=batch_size)
    for batch_counter in range(batch_generator.num_batches):
      images_batch, labels_batch = batch_generator.next()
      loss = one_training_step(model, images_batch, labels_batch)
      if batch_counter % 100 == 0:
        print(
            f"loss at epoch {epoch_counter} batch {batch_counter}: {loss:.2f}")
    if test_fn is not None:
      test_fn()


# main
naive_model = NaiveSequential([
    NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
    NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax)
])
assert len(naive_model.weights) == 4

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

fit(naive_model, train_images, train_labels, batch_size=128,
    test_fn=lambda: test(naive_model, test_images, test_labels))
