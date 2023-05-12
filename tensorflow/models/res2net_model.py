from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .models import gelu, batch_norm, fixed_padding, conv2d_fixed_padding, dense, squeeze_and_excititation, l2_regularizer, stats_pool, att_stats_pool

DEFAULT_VERSION = 1


def res2net_pad_conv_bn_relu(inputs, filters, kernel_size, strides, training, data_format, dilation_rate, trainable, split, width):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    with tf.variable_scope(name_or_scope=None, default_name='conv2d') as scope:
        in_shape = inputs.get_shape().as_list()
        channel_axis = 3 if data_format == 'channels_last' else 1
        in_channels = in_shape[channel_axis]
        out_channels = filters
        assert in_channels % split == 0
        assert out_channels % split == 0
        padding=('SAME' if strides == 1 else 'VALID')
        data_format = 'NHWC' if data_format == 'channels_last' else 'NCHW'
        dilations = dilation_rate
        if type(kernel_size) == int:
            kernel_size = [kernel_size, kernel_size]
        else:
            kernel_size = list(kernel_size)

        group_filters = tf.get_variable(
            name='kernel',
            initializer=tf.compat.v1.variance_scaling_initializer(),
            regularizer=l2_regularizer(1e-3),
            shape=kernel_size + [width, width * (split - 1)],
            trainable=trainable,
            dtype=tf.float32)

        inputs = tf.split(inputs, split, channel_axis)
        kernels = tf.split(group_filters, split - 1, 3)

        outputs = [
            tf.nn.relu(
                batch_norm(
                    inputs=tf.nn.conv2d(inputs[0], kernels[0], strides=strides, padding=padding, data_format=data_format, dilations=dilations),
                    training=training, data_format=data_format))]
        
        for idx in range(1, split - 1):
            input = inputs[idx]
            kernel = kernels[idx]
            if strides == 1:
                input += outputs[idx - 1]

            outputs.append(
                tf.nn.relu(
                    batch_norm(
                        inputs=tf.nn.conv2d(input, kernel, strides=strides, padding=padding, data_format=data_format, dilations=dilations),
                        training=training, data_format=data_format)))

        if strides == 1:
            outputs.append(inputs[split-1])
        else:
            outputs.append(tf.nn.avg_pool2d(inputs[split-1], ksize=3, strides=strides, padding=padding, data_format=data_format))
    return tf.concat(outputs, channel_axis)


def _bottleneck_block_v1(inputs, filters, projection_shortcut,
                         strides, training, data_format, cardinality, use_se, split, width):
    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(shortcut, training=training, data_format=data_format)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=split * width, kernel_size=1, strides=1, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.relu(inputs)

    inputs = res2net_pad_conv_bn_relu(inputs=inputs, filters=split * width, kernel_size=3, strides=strides, training=training,
                                      data_format=data_format, dilation_rate=(1,1), trainable=True, split=split, width=width)

    filters_out = filters * 4 if cardinality == 1 else filters * 2

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters_out, kernel_size=1, strides=1, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                name, training, data_format, cardinality, use_se, split, width):

    # Bottleneck blocks end with 4 times the number of filters as they start with
    # filters_out = filters * 4 if bottleneck else filters
    if bottleneck:
        if cardinality == 1:
            filters_out = filters * 4
        else:
            filters_out = filters * 2
    else:
        filters_out = filters

    def projection_shortcut(inputs):
        # inputs = tf.nn.avg_pool2d(inputs, ksize=strides, strides=strides, padding='SAME',
        #                           data_format=('NHWC' if data_format == 'channels_last' else 'NCHW'))
        # return conv2d_fixed_padding(
        #     inputs=inputs, filters=filters_out, kernel_size=1, strides=1,
        #     data_format=data_format)
        return conv2d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
            data_format=data_format)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, projection_shortcut, strides, training,
                      data_format, cardinality, use_se, split, width)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, None, 1, training, data_format, cardinality, use_se, split, width)

    return tf.identity(inputs, name)


class Model(object):
    """Base class for building the Resnet Model."""

    def __init__(self, resnet_size, bottleneck,
                 num_filters, output_dim, kernel_size, conv_stride, first_pool_size, first_pool_stride,
                 block_sizes, block_strides, resnet_version=DEFAULT_VERSION, data_format='channels_last',
                 cardinality=1, use_se=False, split=4, width=24, temporal_pool=stats_pool):
        self.resnet_size = resnet_size

        # if not data_format:
        #   data_format = ('channels_first' if tf.config.list_physical_devices('GPU')
        #                  else 'channels_last')

        self.resnet_version = resnet_version
        if resnet_version not in (1, 2):
            raise ValueError(
                'Resnet version should be 1 or 2. See README for citations.')

        self.bottleneck = bottleneck
        if bottleneck:
            if resnet_version == 1:
                self.block_fn = _bottleneck_block_v1
            else:
                raise ValueError
        else:
            if resnet_version == 1:
                self.block_fn = _building_block_v1
            else:
                raise ValueError

        self.data_format = data_format
        self.cardinality = cardinality
        self.use_se = use_se
        self.split = split
        self.width = width
        self.num_filters = num_filters
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.first_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride
        self.block_sizes = block_sizes
        self.block_strides = block_strides
        self.pre_activation = resnet_version == 2
        self.temporal_pool = temporal_pool

    def __call__(self, inputs, training):
        if self.data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            inputs = tf.transpose(a=inputs, perm=[0, 3, 1, 2])

        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=self.num_filters[0], kernel_size=self.kernel_size,
            strides=self.conv_stride, data_format=self.data_format)
        inputs = tf.identity(inputs, 'initial_conv')

        # We do not include batch normalization or activation functions in V2
        # for the initial conv1 because the first ResNet unit will perform these
        # for both the shortcut and non-shortcut paths as part of the first
        # block's projection. Cf. Appendix of [2].
        if self.resnet_version == 1:
            inputs = batch_norm(inputs, training=training, data_format=self.data_format)
            inputs = tf.nn.relu(inputs)

        if self.first_pool_size:
            inputs = tf.compat.v1.layers.max_pooling2d(
                inputs=inputs, pool_size=self.first_pool_size,
                strides=self.first_pool_stride, padding='SAME',
                data_format=self.data_format)
        inputs = tf.identity(inputs, 'initial_max_pool')

        for i, num_blocks in enumerate(self.block_sizes):
            # num_filters = self.num_filters * (2**i)
            num_filters = self.num_filters[i]
            # width = self.width * (2**i)
            width = self.width[i]
            inputs = block_layer(
                inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
                block_fn=self.block_fn, blocks=num_blocks,
                strides=self.block_strides[i], name='block_layer{}'.format(i + 1), training=training,
                data_format=self.data_format, cardinality=self.cardinality, use_se=self.use_se, split=self.split, width=width)

        # Only apply the BN and ReLU for model that does pre_activation in each
        # building/bottleneck block, eg resnet V2.
        if self.pre_activation:
            inputs = batch_norm(inputs, training=training, data_format=self.data_format)
            inputs = tf.nn.relu(inputs)

        inputs = self.temporal_pool(inputs, data_format=self.data_format)
        inputs = tf.compat.v1.layers.flatten(inputs, data_format=self.data_format)

        # norm here
        # inputs = batch_norm(inputs, training=training, data_format='channels_first')
        # inputs = dense(inputs, self.output_dim)
        # inputs = tf.identity(inputs, 'final_dense')
        # inputs = batch_norm(inputs, training=training, data_format='channels_first')
        # inputs = tf.identity(inputs, 'final_norm')

        inputs = batch_norm(inputs, training=training, data_format='channels_first')                                                        
        inputs = dense(inputs, self.output_dim)
        inputs = batch_norm(inputs, training=training, data_format='channels_first')
        inputs = tf.identity(inputs, 'outputs')
        return inputs


res2net50_w24_s4_c64 = Model(resnet_size=50, bottleneck=True, num_filters=[64, 128, 256, 512], output_dim=256,
                             kernel_size=3, conv_stride=1, first_pool_size=None, first_pool_stride=None,
                             block_sizes=[3,4,6,3], block_strides=[1,2,2,2],
                             resnet_version=1, data_format='channels_last', split=4, width=[24, 48, 96, 192],
                             temporal_pool=stats_pool)

res2net50_w24_s4_c32 = Model(resnet_size=50, bottleneck=True, num_filters=[32, 64, 128, 256], output_dim=256,
                             kernel_size=3, conv_stride=1, first_pool_size=None, first_pool_stride=None,
                             block_sizes=[3,4,6,3], block_strides=[1,2,2,2],
                             resnet_version=1, data_format='channels_last', split=4, width=[24, 48, 96, 192],
                             temporal_pool=stats_pool)

res2net101_w24_s4_c32_att = Model(resnet_size=101, bottleneck=True, num_filters=[32, 64, 128, 256], output_dim=256,
                                  kernel_size=3, conv_stride=1, first_pool_size=None, first_pool_stride=None,
                                  block_sizes=[3,4,23,3], block_strides=[1,2,2,2],
                                  resnet_version=1, data_format='channels_last', split=4, width=[24, 48, 96, 192],
                                  temporal_pool=att_stats_pool)

res2net152_w24_s4_c32_att = Model(resnet_size=152, bottleneck=True, num_filters=[32, 64, 128, 256], output_dim=256,
                                  kernel_size=3, conv_stride=1, first_pool_size=None, first_pool_stride=None,
                                  block_sizes=[3,8,36,3], block_strides=[1,2,2,2],
                                  resnet_version=1, data_format='channels_last', split=4, width=[24, 48, 96, 192],
                                  temporal_pool=att_stats_pool)

res2net200_w24_s4_c32_att = Model(resnet_size=200, bottleneck=True, num_filters=[32, 64, 128, 256], output_dim=256,
                                  kernel_size=3, conv_stride=1, first_pool_size=None, first_pool_stride=None,
                                  block_sizes=[3,24,36,3], block_strides=[1,2,2,2],
                                  resnet_version=1, data_format='channels_last', split=4, width=[24, 48, 96, 192],
                                  temporal_pool=att_stats_pool)
