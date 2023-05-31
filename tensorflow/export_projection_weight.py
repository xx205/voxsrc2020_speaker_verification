# Copyright (c) 2023 Xu Xiang
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


import argparse
import pickle

import numpy as np
import tensorflow as tf


def l2norm(x, axis=0, keepdims=True):
    x_norm = np.linalg.norm(x, axis=axis, keepdims=keepdims)
    return x / x_norm


def load(checkpoint_path, var_name):
    reader = tf.train.NewCheckpointReader(checkpoint_path)
    var = reader.get_variable_to_shape_map()
    weight = reader.get_tensor('{}'.format(var_name))
    weight = np.swapaxes(weight, -1, -2)
    weight = np.reshape(weight, (-1, weight.shape[-1]))
    weight = l2norm(weight, axis=1)
    return weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_checkpoint", type=str, dest="input_checkpoint",
                        help="tensorflow checkpoint path")
    parser.add_argument("--var_name", type=str, dest="var_name",
                        help="variable name")
    parser.add_argument("--output_weight", type=str, dest="output_weight",
                        help="output weight file")
    args = parser.parse_args()

    weight = load(args.input_checkpoint, args.var_name)
    pickle.dump(weight, open(args.output_weight, 'wb'))

if __name__ == "__main__":
    main()
