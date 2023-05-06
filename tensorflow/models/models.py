import tensorflow as tf
import horovod.tensorflow as hvd

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

import numbers
from tensorflow.python.platform import tf_logging as logging


def l2_regularizer(scale, scope=None):
  """Returns a function that can be used to apply L2 regularization to weights.
  Small values of L2 can help prevent overfitting the training data.
  Args:
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
    scope: An optional scope name.
  Returns:
    A function with signature `l2(weights)` that applies L2 regularization.
  Raises:
    ValueError: If scale is negative or if scale is not a float.
  """
  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % (scale,))
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g.' %
                       scale)
    if scale == 0.:
      logging.info('Scale of 0 disables regularizer.')
      return lambda _: None

  def l2(weights):
    """Applies l2 regularization to weights."""
    with tf.name_scope(scope, 'l2_regularizer', [weights]) as name:
      my_scale = tf.convert_to_tensor(scale,
                                      dtype=weights.dtype.base_dtype,
                                      name='scale')
      return tf.multiply(my_scale, tf.nn.l2_loss(weights), name=name)

  return l2


def gelu(inputs):
  return inputs * 0.5 * (1.0 + tf.erf(inputs / tf.sqrt(2.0)))


def batch_norm(inputs, training, data_format):
  """Performs a batch normalization using a standard set of parameters."""
  return tf.compat.v1.layers.batch_normalization(
    inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
    momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=False,
    scale=False, training=training, fused=True)


# def sync_batch_norm(inputs, training, data_format):
#   return hvd.SyncBatchNormalization(axis=1 if data_format == 'channels_first' else 3,
#                                     momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=False,
#                                     scale=False, fused=False)(inputs=inputs, training=training)

# Specify the variable scope options
# def layer_norm(inputs, axes=[-1]):
#   with tf.variable_scope(name_or_scope=None, default_name='layer_norm') as scope:
#     mean, var = tf.nn.moments(inputs, axes=axes, keep_dims=True)
#     return (inputs - mean) * tf.rsqrt(var + 1e-5)


# Rewrite
# Specify the variable scope options
def layer_norm(inputs, axes=[-1], reuse=None, scope=None, data_format='channels_last'):
  with tf.variable_scope(name_or_scope=scope, default_name="layer_norm", values=[inputs], reuse=reuse):
    shape = inputs.get_shape().as_list()
    ndims = len(shape)
    assert ndims in [2, 4]

    # if data_format == 'channels_last':
    #   data_format = 'NHWC'
    #   channels = shape[-1]
    # elif data_format == 'channels_first':
    #   data_format = 'NCHW'
    #   channels = shape[1]
    # else:
    #   raise ValueError

    mean, var = tf.nn.moments(inputs, axes, keep_dims=True)
    beta = tf.zeros([1] * ndims, name='beta')
    gamma = tf.ones([1] * ndims, name='gamma')

    inputs = tf.compat.v1.nn.batch_normalization(inputs, mean, var, beta, gamma, _BATCH_NORM_EPSILON)
    return inputs


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  if isinstance(kernel_size, int):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
      padded_inputs = tf.pad(tensor=inputs,
                             paddings=[[0, 0], [0, 0], [pad_beg, pad_end],
                                       [pad_beg, pad_end]])
    else:
      padded_inputs = tf.pad(tensor=inputs,
                             paddings=[[0, 0], [pad_beg, pad_end],
                                       [pad_beg, pad_end], [0, 0]])
    return padded_inputs
  else:
    pad_total_h = kernel_size[0] - 1
    pad_beg_h = pad_total_h // 2
    pad_end_h = pad_total_h - pad_beg_h

    pad_total_w = kernel_size[1] - 1
    pad_beg_w = pad_total_w // 2
    pad_end_w = pad_total_w - pad_beg_w

    if data_format == 'channels_first':
      padded_inputs = tf.pad(tensor=inputs,
                             paddings=[[0, 0], [0, 0], [pad_beg_h, pad_end_h],
                                       [pad_beg_w, pad_end_w]])
    else:
      padded_inputs = tf.pad(tensor=inputs,
                             paddings=[[0, 0], [pad_beg_h, pad_end_h],
                                       [pad_beg_w, pad_end_w], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format, cardinality=1, l2=1e-3):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

  padding = 'SAME'
  if (isinstance(strides, int) and strides > 1) or (not isinstance(strides, int) and any(s > 1 for s in strides)):
  # if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)
    padding = 'VALID'

  return conv2d(
    inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
    padding=padding, data_format=data_format, cardinality=cardinality, l2=l2)
    # padding=('SAME' if strides == 1 else 'VALID'), data_format=data_format, cardinality=cardinality)


# Specify the variable scope options
def conv2d(inputs, filters, kernel_size, strides=1, padding='valid', dilation_rate=(1,1), trainable=True, cardinality=1, reuse=None, scope=None, data_format='channels_last', l2=1e-3):
  with tf.variable_scope(name_or_scope=scope, default_name='conv2d', values=[inputs], reuse=reuse):
    in_shape = inputs.get_shape().as_list()
    channel_axis = 3 if data_format == 'channels_last' else 1
    in_channels = in_shape[channel_axis]
    out_channels = filters

    assert in_channels % cardinality == 0
    assert out_channels % cardinality == 0

    padding = padding.upper() if isinstance(padding, str) else padding
    data_format = 'NHWC' if data_format == 'channels_last' else 'NCHW'
    dilations = dilation_rate
    if type(kernel_size) == int:
      kernel_size = [kernel_size, kernel_size]
    else:
      kernel_size = list(kernel_size)

    group_filters = tf.get_variable(
      name='kernel',
      initializer=tf.compat.v1.variance_scaling_initializer(),
      regularizer=l2_regularizer(l2),
      shape=kernel_size + [in_channels // cardinality, out_channels],
      trainable=trainable,
      dtype=tf.float32)
    return tf.nn.conv2d(inputs,
                        group_filters,
                        strides=strides,
                        padding=padding,
                        data_format=data_format,
                        dilations=dilations)

  # For running on CPU
  # if cardinality == 1:
  #   return tf.nn.conv2d(inputs,
  #                       group_filters,
  #                       strides=strides,
  #                       padding=padding,
  #                       data_format=data_format,
  #                       dilations=dilations)
  # else:
  #   inputs = tf.split(inputs, cardinality, channel_axis)
  #   kernels = tf.split(group_filters, cardinality, 3)
  #   outputs = [tf.nn.conv2d(i, k, strides=strides, padding=padding, data_format=data_format, dilations=dilations)
  #              for i, k in zip(inputs, kernels)]
  #   return tf.concat(outputs, channel_axis)


# def conv2d_general(inputs, filters, kernel_size, strides=1, padding='valid', data_format='channels_last',
#                    dilation_rate=(1,1), trainable=True, cardinality=1):
#   with tf.variable_scope(name_or_scope=None, default_name='conv2d') as scope:
#     in_shape = inputs.get_shape().as_list()
#     channel_axis = 3 if data_format == 'channels_last' else 1
#     in_channels = in_shape[channel_axis]
#     out_channels = filters
#     assert in_channels % cardinality == 0
#     assert out_channels % cardinality == 0
#     padding = padding.upper() if isinstance(padding, str) else padding
#     data_format = 'NHWC' if data_format == 'channels_last' else 'NCHW'
#     dilations = dilation_rate
#     if type(kernel_size) == int:
#       kernel_size = [kernel_size, kernel_size]
#     else:
#       kernel_size = list(kernel_size)

#     group_filters = tf.get_variable(
#       name='kernel',
#       initializer=tf.compat.v1.variance_scaling_initializer(),
#       regularizer=l2_regularizer(1e-3),
#       shape=kernel_size + [in_channels // cardinality, out_channels],
#       trainable=trainable,
#       dtype=tf.float32)
#     # For running on CPU
#     if cardinality == 1:
#       return tf.nn.conv2d(inputs,
#                           group_filters,
#                           strides=strides,
#                           padding=padding,
#                           data_format=data_format,
#                           dilations=dilations)
#     else:
#       inputs = tf.split(inputs, cardinality, channel_axis)
#       kernels = tf.split(group_filters, cardinality, 3)
#       outputs = [tf.nn.conv2d(i, k, strides=strides, padding=padding, data_format=data_format, dilations=dilations)
#                  for i, k in zip(inputs, kernels)]
#       return tf.concat(outputs, channel_axis)


# Specify the variable scope options
def stats_pool(inputs, epsilon=1e-5, reuse=None, scope=None, data_format='channels_last'):
  with tf.variable_scope(name_or_scope=scope, default_name='stats_pool', values=[inputs], reuse=reuse):
    # N H W C    Reduce operation on H
    # 0 1 2 3    Concatenate operation on C
    axis = (1 if data_format == 'channels_last' else 2)
    mean, var = tf.nn.moments(inputs, axes=[axis], keep_dims=True)
    mean_std = tf.concat([mean, tf.sqrt(var + epsilon)], 3)
    return mean_std


# Specify the variable scope options
def att_stats_pool(inputs, att_dim=128, epsilon=1e-5, trainable=True, att_with_mean_std=True, reuse=None, scope=None, data_format='channels_last'):
  with tf.variable_scope(name_or_scope=scope, default_name='att_stats_pool', values=[inputs], reuse=reuse):
    time_axis = (1 if data_format == 'channels_last' else 2)
    channel_axis = (3 if data_format == 'channels_last' else 1)
    num_channels = inputs.get_shape().as_list()[channel_axis]

    mean, var = tf.nn.moments(inputs, axes=[time_axis], keep_dims=True)

    if att_with_mean_std:
      mean_std = tf.concat([mean, tf.sqrt(var + epsilon)], channel_axis)
      # NHWC
      # multiples = tf.raw_ops.Pack(values=[1, tf.shape(inputs)[time_axis], 1, 1])
      # NCHW
      # multiples = tf.raw_ops.Pack(values=[1, 1, tf.shape(inputs)[time_axis], 1])
      values = [1, 1, 1]
      values.insert(time_axis, tf.shape(inputs)[time_axis])
      multiples = tf.raw_ops.Pack(values=values)
      mean_std = tf.tile(mean_std, multiples)
      att_inputs = tf.concat([inputs, mean_std], axis=channel_axis)
    else:
      att_inputs = inputs

    weights = tf.nn.softmax(
      conv2d(
        tf.nn.tanh(conv2d(att_inputs, filters=att_dim, kernel_size=1, data_format=data_format)),
        filters=num_channels, kernel_size=1, data_format=data_format), axis=time_axis)

    weighted_mean = tf.reduce_sum(inputs * weights, axis=time_axis, keep_dims=True)
    weighted_sum_square = tf.reduce_sum(inputs * inputs * weights, axis=time_axis, keep_dims=True)
    weighted_std = tf.sqrt(weighted_sum_square - weighted_mean * weighted_mean + epsilon)
  return tf.concat([weighted_mean, weighted_std], axis=channel_axis)


def dense(inputs, units, l2=1e-3):
  return tf.compat.v1.layers.dense(inputs=inputs, units=units, use_bias=False,
                                   kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                                   kernel_regularizer=l2_regularizer(l2))


# Specify the variable scope options
def squeeze_and_excititation(inputs, ratio=16, reuse=None, scope=None, data_format='channels_last'):
  with tf.variable_scope(name_or_scope=scope, default_name='squeeze_and_excititation', values=[inputs], reuse=reuse):
    in_shape = inputs.get_shape().as_list()
    channel_axis = (3 if data_format == 'channels_last' else 1)
    in_channels = in_shape[channel_axis]
    assert in_channels % ratio == 0
    axis = ([1,2] if data_format == 'channels_last' else [2,3])

    scale = tf.reduce_mean(inputs, axis=axis, keep_dims=True)
    scale = conv2d(scale, filters=in_channels // ratio, kernel_size=1, data_format=data_format)
    scale = tf.nn.relu(scale)
    scale = conv2d(scale, filters=in_channels, kernel_size=1, data_format=data_format)
    scale = tf.nn.sigmoid(scale)
    return scale * inputs
