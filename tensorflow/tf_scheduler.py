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


import math
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops


# def triangular2(max_learning_rate, min_learning_rate, global_step, num_cycles, cycle_length, name=None):
#     with ops.name_scope(name, "Triangular2", [max_learning_rate, min_learning_rate, global_step, num_cycles, cycle_length]) as name:
#         max_learning_rate = ops.convert_to_tensor(max_learning_rate, name="max_learning_rate")
#         dtype = max_learning_rate.dtype
#         min_learning_rate = ops.convert_to_tensor(min_learning_rate, name="min_learning_rate")
#         global_step = math_ops.cast(global_step, dtype)
#         num_cycles = math_ops.cast(num_cycles, dtype)
#         cycle_length = math_ops.cast(cycle_length, dtype)

#         q = global_step % cycle_length
#         r = global_step // cycle_length
#         s = q / (cycle_length / 2.0)
#         t = math_ops.pow(0.5, r)

#         pred_fn_pairs = []
#         pred_fn_pairs.append((q <= cycle_length / 2,
#                               lambda: t * (min_learning_rate + (max_learning_rate - min_learning_rate) * s)))
#         pred_fn_pairs.append((q > cycle_length / 2,
#                               lambda: t * (min_learning_rate + (max_learning_rate - min_learning_rate) * (2 - s))))
#         # The default isn't needed here because our conditions are mutually
#         # exclusive and exhaustive, but tf.case requires it.
#         default = lambda: 0.0
#         return control_flow_ops.case(pred_fn_pairs, default, exclusive=True)


def warmup_constant_exponential_decay(learning_rate, global_step, boundaries, decay_steps, decay_rate=0.5, staircase=True, name=None):
    if global_step is None:
        raise ValueError("global_step is required for warmup_constant_exponential_decay.")
    if len(boundaries) != 3:
        raise ValueError("The length of boundaries should be 3.")
    with ops.name_scope(name, "WarmupConstantExponentialDecay", [learning_rate, global_step, boundaries, decay_steps, decay_rate]) as name:
        learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        global_step = math_ops.cast(global_step, dtype)
        boundaries = ops.convert_n_to_tensor(boundaries, dtype)
        decay_steps = math_ops.cast(decay_steps, dtype)
        decay_rate = math_ops.cast(decay_rate, dtype)
        q = global_step / boundaries[0]
        p = (global_step - boundaries[1]) / decay_steps
        if staircase:
            p = math_ops.ceil(p)
        pred_fn_pairs = []
        pred_fn_pairs.append((global_step <= boundaries[0],
                              lambda: math_ops.multiply(learning_rate, q)))
        pred_fn_pairs.append(((global_step > boundaries[0]) & (global_step <= boundaries[1]),
                              lambda: learning_rate))
        pred_fn_pairs.append(((global_step > boundaries[1]) & (global_step <= boundaries[2]),
                              lambda: math_ops.multiply(learning_rate, math_ops.pow(decay_rate, p))))
        # Finetuning learning rate
        pred_fn_pairs.append((global_step > boundaries[2],
                              lambda: math_ops.multiply(learning_rate, math_ops.cast(1.0 / 128, dtype))))        
        # The default isn't needed here because our conditions are mutually
        # exclusive and exhaustive, but tf.case requires it.
        default = lambda: 0.0
        return control_flow_ops.case(pred_fn_pairs, default, exclusive=True)


def warmup_constant_cosine_decay(learning_rate, global_step, boundaries, name=None):
    if global_step is None:
        raise ValueError("global_step is required for warmup_constant_cosine_decay.")
    if len(boundaries) != 3:
        raise ValueError("The length of boundaries should be 3.")
    with ops.name_scope(name, "WarmupConstantCosineDecay", [learning_rate, global_step, boundaries]) as name:
        learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        global_step = math_ops.cast(global_step, dtype)
        boundaries = ops.convert_n_to_tensor(boundaries, dtype)
        q = global_step / boundaries[0]
        p = (global_step - boundaries[1]) / (boundaries[2] - boundaries[1])
        pred_fn_pairs = []
        pred_fn_pairs.append((global_step <= boundaries[0],
                              lambda: math_ops.multiply(learning_rate, q)))
        pred_fn_pairs.append(((global_step > boundaries[0]) & (global_step <= boundaries[1]),
                              lambda: learning_rate))
        pred_fn_pairs.append(((global_step > boundaries[1]) & (global_step <= boundaries[2]),
                              lambda: math_ops.multiply(learning_rate, 0.5 * (1.0 + math_ops.cos(p * math.pi)))))
        # Finetuning learing rate
        pred_fn_pairs.append((global_step > boundaries[2],
                              lambda: math_ops.multiply(learning_rate, math_ops.cast(1.0 / 128, dtype))))
        # The default isn't needed here because our conditions are mutually
        # exclusive and exhaustive, but tf.case requires it.
        default = lambda: 0.0
        return control_flow_ops.case(pred_fn_pairs, default, exclusive=True)


def zero_linear_constant(margin, global_step, boundaries, grow_steps, staircase=True, name='None'):
    if global_step is None:
        raise ValueError("global_step is required for margin.")
    if len(boundaries) != 2:
        raise ValueError("The length of boundaries should be 2.")
    with ops.name_scope(name, "ZeroLinearConstant", [margin, global_step, boundaries, grow_steps]) as name:
        margin = ops.convert_to_tensor(margin, name="margin")
        dtype = margin.dtype
        global_step = math_ops.cast(global_step, dtype)
        boundaries = ops.convert_n_to_tensor(boundaries, dtype)

        p = (global_step - boundaries[0]) / grow_steps
        if staircase:
            p = math_ops.ceil(p)

        pred_fn_pairs = []
        pred_fn_pairs.append((global_step <= boundaries[0],
                              lambda: 0.0 * margin))
        pred_fn_pairs.append(((global_step > boundaries[0]) & (global_step <= boundaries[1]),
                              lambda: math_ops.multiply(margin, math_ops.divide(p * grow_steps, (boundaries[1] - boundaries[0])))))
        pred_fn_pairs.append((global_step > boundaries[1],
                              lambda: margin))
        # The default isn't needed here because our conditions are mutually
        # exclusive and exhaustive, but tf.case requires it.
        default = lambda: 0.0
        return control_flow_ops.case(pred_fn_pairs, default, exclusive=True)
