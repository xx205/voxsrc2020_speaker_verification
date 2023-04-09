from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .models import gelu, batch_norm, layer_norm, stats_pool, att_stats_pool, conv2d, dense, squeeze_and_excititation, l2_regularizer


def conv_relu_bn_block(inputs, filters, kernel_size, padding, data_format, dilation_rate, cardinality, training, name):
  inputs = conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, padding=padding,
      data_format=data_format, dilation_rate=dilation_rate, cardinality=cardinality)
  inputs = tf.nn.relu(inputs)
  inputs = batch_norm(inputs, training, data_format)
  return tf.identity(inputs, name)


def conv_relu_ln_block(inputs, filters, kernel_size, padding, data_format, dilation_rate, cardinality, training, name):
  inputs = conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, padding=padding,
      data_format=data_format, dilation_rate=dilation_rate, cardinality=cardinality)
  inputs = tf.nn.relu(inputs)
  inputs = layer_norm(inputs)
  return tf.identity(inputs, name)


def conv_gelu_bn_block(inputs, filters, kernel_size, padding, data_format, dilation_rate, cardinality, training, name):
  inputs = conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, padding=padding,
      data_format=data_format, dilation_rate=dilation_rate, cardinality=cardinality)
  inputs = gelu(inputs)
  inputs = batch_norm(inputs, training, data_format)
  return tf.identity(inputs, name)


def conv_gelu_ln_block(inputs, filters, kernel_size, padding, data_format, dilation_rate, cardinality, training, name):
  inputs = conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, padding=padding,
      data_format=data_format, dilation_rate=dilation_rate, cardinality=cardinality)
  inputs = gelu(inputs)
  inputs = layer_norm(inputs)
  return tf.identity(inputs, name)


def conv_se_relu_bn_block(inputs, filters, kernel_size, padding, data_format, dilation_rate, cardinality, training, name):
  inputs = conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, padding=padding,
      data_format=data_format, dilation_rate=dilation_rate, cardinality=cardinality)
  inputs = squeeze_and_excititation(inputs, ratio=16, data_format=data_format)
  inputs = tf.nn.relu(inputs)
  inputs = batch_norm(inputs, training, data_format)
  return tf.identity(inputs, name)


def conv_relu_se_bn_block(inputs, filters, kernel_size, padding, data_format, dilation_rate, cardinality, training, name):
  inputs = conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, padding=padding,
      data_format=data_format, dilation_rate=dilation_rate, cardinality=cardinality)
  inputs = tf.nn.relu(inputs)
  inputs = squeeze_and_excititation(inputs, ratio=8, data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  return tf.identity(inputs, name)


def conv_mish_bn_block(inputs, filters, kernel_size, padding, data_format, dilation_rate, cardinality, training, name):
  inputs = conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, padding=padding,
      data_format=data_format, dilation_rate=dilation_rate, cardinality=cardinality)
  inputs = inputs * tf.tanh(tf.nn.softplus(inputs))
  inputs = batch_norm(inputs, training, data_format)
  return tf.identity(inputs, name)


def mish(inputs):
  return inputs * tf.tanh(tf.nn.softplus(inputs))


def conv_bn_relu_block(inputs, filters, kernel_size, padding, data_format, dilation_rate, cardinality, training, name):
  inputs = conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, padding=padding,
      data_format=data_format, dilation_rate=dilation_rate, cardinality=cardinality)
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)
  return tf.identity(inputs, name)


def conv_bn_se_relu_block(inputs, filters, kernel_size, padding, data_format, dilation_rate, cardinality, training, name):
  inputs = conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, padding=padding,
      data_format=data_format, dilation_rate=dilation_rate, cardinality=cardinality)
  inputs = batch_norm(inputs, training, data_format)
  inputs = squeeze_and_excititation(inputs, ratio=16, data_format=data_format)
  inputs = tf.nn.relu(inputs)
  return tf.identity(inputs, name)


# TODO: use conv instead of dense to prevent the use of flatten?
# TODO: check the equivalence of the two implementations
class Model(object):
  def __init__(self, output_dim, padding='valid', block_fn=None, block_filters=None, block_kernel_sizes=None,
               block_dilations=None, block_cardinalities=None, data_format='channels_last', time_pool=stats_pool):
    self.output_dim = output_dim
    self.padding = padding
    self.block_fn = block_fn
    self.block_filters = block_filters
    self.block_kernel_sizes = block_kernel_sizes
    self.block_dilations = block_dilations
    self.block_cardinalities = block_cardinalities
    self.data_format = data_format
    self.time_pool = time_pool
    # self.act = act

  def __call__(self, inputs, training):
    if self.data_format == 'channels_first':
      # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
      # This provides a large performance boost on GPU. See
      # https://www.tensorflow.org/performance/performance_guide#data_formats
      inputs = tf.transpose(a=inputs, perm=[0, 3, 1, 2])

    for i in range(len(self.block_kernel_sizes)):
      cardinality = (1 if self.block_cardinalities is None else self.block_cardinalities[i])
      dilation_rate = (1 if self.block_dilations is None else self.block_dilations[i])
      inputs = self.block_fn(
          inputs=inputs, filters=self.block_filters[i], kernel_size=self.block_kernel_sizes[i], padding=self.padding,
          data_format=self.data_format, dilation_rate=dilation_rate, cardinality=cardinality, training=training,
          name='block{}'.format(i + 1))#, act=self.act)
    inputs = self.time_pool(inputs, data_format=self.data_format)
    inputs = tf.compat.v1.layers.flatten(inputs, data_format=self.data_format)
    # inputs = dense(inputs, self.output_dim)
    # inputs = tf.identity(inputs, 'outputs')

    # For two dimensional input data: N, C, data_format is set to 'channels_first'
    inputs = batch_norm(inputs, training, 'channels_first')
    inputs = dense(inputs, self.output_dim)
    # inputs = tf.identity(inputs, 'final_dense')
    inputs = batch_norm(inputs, training, 'channels_first')
    # inputs = tf.identity(inputs, 'final_norm')
    inputs = tf.identity(inputs, 'outputs')

    return inputs


tdnn = Model(output_dim=256, padding='same', block_fn=conv_relu_bn_block, block_filters=[512, 512, 512, 512, 1536],
             block_kernel_sizes=[(5,1), (3,1), (3,1), (1,1), (1,1)],
             block_dilations=[(1,1), (2,1), (3,1), (1,1), (1,1)],
             block_cardinalities=None, data_format='channels_last')
