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


import tensorflow as tf
import numpy as np
import sys
import os
import kaldi_io
import multiprocessing as mp
import argparse
import pickle


def get_batch(queue, scp, BATCH_SIZE, expand_dim):
    assert BATCH_SIZE == 1
    test_data_generator = kaldi_io.read_mat_ark(scp)

    feature_list, label_list = [], []
    for label, feature in test_data_generator:
        feature = np.expand_dims(feature, expand_dim - 1)
        if len(feature_list) == BATCH_SIZE:
            queue.put((np.stack(feature_list), label_list))
            feature_list, label_list = [feature], [label]
        else:
            feature_list.append(feature)
            label_list.append(label)
    if len(feature_list) > 0:
        queue.put((np.stack(feature_list), label_list))
    # No more data
    queue.put((None, None))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pb-file', action='store', dest='pb_file', default='',
                        help='protobuffer model file')
    parser.add_argument('--expand-dim', action='store', dest='expand_dim', default=2, type=int,
                        help='expansion dimension of the input feature')
    parser.add_argument('--rspec', action='store', dest='rspec', default='/tmp/fbank',
                        help='source fbank scp path specification, without ".scp" suffix')
    parser.add_argument('--wspec', action='store', dest='wspec', default='/tmp/xvector',
                        help='destination xvector scp path specification, without ".scp" suffix')
    # parser.add_argument('--scp-base', action='store', dest='scp_base', default='/tmp/feats',
    #                     help='base of scp filename')
    # parser.add_argument('--emb-dir', action='store', dest='emb_dir', default='/tmp',
    #                     help='embedding data directory')
    # parser.add_argument('--rank', action='store', dest='rank', default=-1, type=int,
    #                     help='set the rank of data process')
    args=parser.parse_args()

    rspec = 'ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:{}.scp ark:- |'.format(args.rspec)

    wspec = 'ark:| copy-vector ark:- ark,scp:{0}.ark,{0}.scp'.format(args.wspec)

    # feat_rspec_tpl='ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:{0}/feats.scp ark:- |'
    # scp = args.scp_base + '/' + str(args.rank) + '/'
    # rspec = feat_rspec_tpl.format(scp)
    
    # feat_wspec_tpl = 'ark:| copy-vector ark:- ark,scp:{0}.ark,{0}.scp'
    # emb_scp = os.path.join(args.emb_dir, 'xvector.' + str(args.rank))
    # wspec = feat_wspec_tpl.format(emb_scp)
    
    with open(args.pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="model")
        x_tensor = graph.get_tensor_by_name("model/inputs:0")
        target_tensor = graph.get_tensor_by_name("model/outputs:0")

    BATCH_SIZE = 1
    context = mp.get_context("spawn")
    test_queue = context.Queue(32)
    test_data_process = context.Process(target=get_batch,
                                        args=(test_queue, rspec, BATCH_SIZE, args.expand_dim))
    test_data_process.daemon = True
    test_data_process.start()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    with tf.Session(graph=graph, config=config) as sess:
        with kaldi_io.open_or_fd(wspec, 'wb') as ark_scp:
            max_frames = 1000
            while True:
                x_values, utts = test_queue.get()
                if x_values is None:
                    break
                # the input length is at least 25 frames
                num_chunks = 1 + (x_values.shape[1] - 25) // max_frames

                target_values = []
                input_lengths = []
                for i in range(num_chunks):
                    input_length = (max_frames if (i + 1) * max_frames <= x_values.shape[1] else x_values.shape[1] - i * max_frames)
                    target_value = sess.run(target_tensor, feed_dict={x_tensor: x_values[:, i * max_frames: i * max_frames + input_length, :, :]})
                    target_values.append(target_value * input_length)
                    input_lengths.append(input_length)
                target_values = sum(target_values) / sum(input_lengths)
                for i in range(len(utts)):
                    kaldi_io.write_vec_flt(ark_scp, target_values[i], utts[i])
