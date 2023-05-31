# Copyright (c) 2020 Xu Xiang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .models import batch_norm, conv2d, stats_pool, att_stats_pool, dense, squeeze_and_excititation


def concat_bn_relu(inputs, training, data_format):
    channel_axis = (3 if data_format == 'channels_last' else 1)
    inputs = tf.concat(inputs, channel_axis)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    return inputs


def conv_bn_relu(inputs, filters, kernel_size, strides, training, data_format):
    inputs = conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, padding='same',
                    strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    return inputs


def bn_relu_conv(inputs, filters, kernel_size, strides, training, data_format, cardinality):
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, padding='same',
                    strides=strides, data_format=data_format, cardinality=cardinality)
    return inputs


def bn_relu_conv_layers(inputs, num_1_a, num_3_b, num_1_c, inc, strides, training, data_format, cardinality, use_se):
    inputs = bn_relu_conv(inputs=inputs, filters=num_1_a, kernel_size=1, strides=1, training=training, data_format=data_format, cardinality=1)
    inputs = bn_relu_conv(inputs=inputs, filters=num_3_b, kernel_size=3, strides=strides, training=training, data_format=data_format, cardinality=cardinality)
    if use_se:
        inputs = squeeze_and_excititation(inputs, ratio=8, data_format=data_format)
    inputs = bn_relu_conv(inputs=inputs, filters=num_1_c + inc, kernel_size=1, strides=1, training=training, data_format=data_format, cardinality=1)
    return inputs


def dual_path_block(inputs, num_1_a, num_3_b, num_1_c, inc, projection_type, training, data_format, cardinality, use_se=False, use_basic_block=False):
    channel_axis = (3 if data_format == 'channels_last' else 1)
    strides = 1
    proj = True
    assert projection_type in ['projected', 'downsampled', 'normal']

    if projection_type == 'downsampled':
        strides = 2
    if projection_type == 'normal':
        proj = False

    if proj == False:
        inputs0, inputs1 = inputs[0], inputs[1]
        if isinstance(inputs, list):
            inputs = tf.concat(inputs, channel_axis)
    else:
        if isinstance(inputs, list):
            inputs = tf.concat(inputs, channel_axis)
        projected_inputs = bn_relu_conv(inputs, filters=num_1_c + 2 * inc, kernel_size=1, strides=strides, training=training, data_format=data_format, cardinality=1)
        if data_format == 'channels_last':
            inputs0, inputs1 = projected_inputs[:, :, :, 0:num_1_c], projected_inputs[:, :, :, num_1_c:]
        else:
            inputs0, inputs1 = projected_inputs[:, 0:num_1_c, :, :], projected_inputs[:, num_1_c:, :, :]

    inputs = bn_relu_conv_layers(inputs, num_1_a=num_1_a, num_3_b=num_3_b, num_1_c=num_1_c, inc=inc, strides=strides, training=training, data_format=data_format, cardinality=cardinality, use_se=use_se)

    if data_format == 'channels_last':
        inputs_0, inputs_1 = inputs[:, :, :, 0:num_1_c], inputs[:, :, :, num_1_c:]
    else:
        inputs_0, inputs_1 = inputs[:, 0:num_1_c, :, :], inputs[:, num_1_c:, :, :]
    return [inputs0 + inputs_0, tf.concat([inputs1, inputs_1], channel_axis)]


class Model(object):
    def __init__(self, output_dim, num_init_features, kernel_size=3, strides=1,
                 projection_types=['projected', 'downsampled', 'downsampled', 'downsampled'],
                 bw=64, k_r=128, cardinality=32, k_sec=[3,4,12,3], inc_sec=[16,32,32,64], bw_factor=1,
                 data_format='channels_last', temporal_pool=stats_pool, use_se=False, use_basic_block=False):
        self.output_dim = output_dim
        self.num_init_features = num_init_features
        self.kernel_size = kernel_size
        self.strides = strides
        self.projection_types = projection_types
        self.bw = bw
        self.k_r = k_r
        self.cardinality = cardinality
        self.k_sec = k_sec
        self.inc_sec = inc_sec
        self.bw_factor = bw_factor
        self.data_format = data_format
        self.temporal_pool = temporal_pool
        self.use_se = use_se
        self.use_basic_block = use_basic_block

    def __call__(self, inputs, training):
        inputs = conv_bn_relu(inputs, filters=self.num_init_features, kernel_size=self.kernel_size, strides=self.strides, training=training, data_format=self.data_format)
        inputs = tf.identity(inputs, 'initial_conv')

        bw = self.bw * 1 * self.bw_factor
        bw = int(bw)
        inc = self.inc_sec[0]
        r = self.k_r * bw // (self.bw * self.bw_factor)
        inputs = dual_path_block(inputs, r, r, bw, inc, projection_type=self.projection_types[0], training=training, data_format=self.data_format, cardinality=self.cardinality, use_se=self.use_se, use_basic_block=self.use_basic_block)
        for i in range(1, self.k_sec[0]):
            inputs = dual_path_block(inputs, r, r, bw, inc, projection_type='normal', training=training, data_format=self.data_format, cardinality=self.cardinality, use_se=self.use_se, use_basic_block=self.use_basic_block)
        # inputs = tf.identity(inputs, 'block_layer1')

        bw = self.bw * 2 * self.bw_factor
        bw = int(bw)
        inc = self.inc_sec[1]
        r = self.k_r * bw // (self.bw * self.bw_factor)
        inputs = dual_path_block(inputs, r, r, bw, inc, projection_type=self.projection_types[1], training=training, data_format=self.data_format, cardinality=self.cardinality, use_se=self.use_se, use_basic_block=self.use_basic_block)
        for i in range(1, self.k_sec[1]):
            inputs = dual_path_block(inputs, r, r, bw, inc, projection_type='normal', training=training, data_format=self.data_format, cardinality=self.cardinality, use_se=self.use_se, use_basic_block=self.use_basic_block)
        # inputs = tf.identity(inputs, 'block_layer2')

        bw = self.bw * 4 * self.bw_factor
        bw = int(bw)
        inc = self.inc_sec[2]
        r = self.k_r * bw // (self.bw * self.bw_factor)
        inputs = dual_path_block(inputs, r, r, bw, inc, projection_type=self.projection_types[2], training=training, data_format=self.data_format, cardinality=self.cardinality, use_se=self.use_se, use_basic_block=self.use_basic_block)
        for i in range(1, self.k_sec[2]):
            inputs = dual_path_block(inputs, r, r, bw, inc, projection_type='normal', training=training, data_format=self.data_format, cardinality=self.cardinality, use_se=self.use_se, use_basic_block=self.use_basic_block)
        # inputs = tf.identity(inputs, 'block_layer3')

        bw = self.bw * 8 * self.bw_factor
        bw = int(bw)
        inc = self.inc_sec[3]
        r = self.k_r * bw // (self.bw * self.bw_factor)
        inputs = dual_path_block(inputs, r, r, bw, inc, projection_type=self.projection_types[3], training=training, data_format=self.data_format, cardinality=self.cardinality, use_se=self.use_se, use_basic_block=self.use_basic_block)
        for i in range(1, self.k_sec[3]):
            inputs = dual_path_block(inputs, r, r, bw, inc, projection_type='normal', training=training, data_format=self.data_format, cardinality=self.cardinality, use_se=self.use_se, use_basic_block=self.use_basic_block)
        # inputs = tf.identity(inputs, 'block_layer4')

        inputs = concat_bn_relu(inputs, training=training, data_format=self.data_format)

        inputs = self.temporal_pool(inputs, data_format=self.data_format)
        inputs = tf.compat.v1.layers.flatten(inputs, self.data_format)

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


dpn68 = Model(output_dim=256, num_init_features=10, k_r=128, cardinality=32, k_sec=[3,4,12,3], inc_sec=[16,32,32,64], data_format='channels_last', temporal_pool=stats_pool)
