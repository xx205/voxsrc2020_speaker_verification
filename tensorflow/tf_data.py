#!/usr/bin/env python3
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


import pickle
import numpy as np
import kaldi_io
import spec_augment

class DataGenerator():
    def __init__(self, scp, feat_rspec_tpl, utt2id_pkl, cmvn_pkl, utt2kwd_pkl, num_classes, feat_dim, feat_length, min_feat_length, max_feat_length, training, specaug):
        self.feat_rspec = feat_rspec_tpl.format(scp)
        self.utt2id_pkl = utt2id_pkl
        self.utt2kwd_pkl = utt2kwd_pkl
        self.cmvn_pkl = cmvn_pkl

        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.feat_length = feat_length
        self.min_feat_length = min_feat_length
        self.max_feat_length = max_feat_length
        self.training = training
        self.specaug = specaug

        
        # if self.utt2id_pkl is not None:
        if self.utt2id_pkl != "None":
            self.utt2id = pickle.load(open(self.utt2id_pkl, 'rb'))
        else:
            self.utt2id = None
        # if self.utt2kwd_pkl is not None:
        if self.utt2kwd_pkl != "None":
            self.utt2kwd = pickle.load(open(self.utt2kwd_pkl, 'rb'))
        else:
            self.utt2kwd = None
        # if self.cmvn_pkl is not None:
        if self.cmvn_pkl != "None":
            self.mean, self.std = pickle.load(open(self.cmvn_pkl, 'rb'))
        else:
            self.mean, self.std = None, None

    def __len__(self):
        return np.iinfo(np.dtype('int32')).max

    def __iter__(self):
        generator = kaldi_io.read_mat_ark(self.feat_rspec)

        FEAT_DIM = self.feat_dim
        FEAT_LENGTH = self.feat_length
        MIN_FEAT_LENGTH = self.min_feat_length
        MAX_FEAT_LENGTH = self.max_feat_length

        while True:
            try:
                utt, feat = next(generator)
                if self.training:
                    if np.random.randint(0, 100) >= 90:
                        continue
            
            except StopIteration:
                if self.training:
                    generator = kaldi_io.read_mat_ark(self.feat_rspec)
                    utt, feat = next(generator)
                else:
                    break

            # Normalize feat first
            if self.mean is not None and self.std is not None:
                feat = (feat - self.mean) / self.std

            # Data augmentation during training
            if self.training:
                FEAT_LENGTH_ = FEAT_LENGTH

                t = np.zeros((FEAT_LENGTH_, FEAT_DIM), dtype=np.float32)
                # If len(feat) < FEAT_LENGTH_, randomly put the feat into data of length of FEAT_LENGTH_
                if feat.shape[0] < FEAT_LENGTH_:
                    random_shift = np.random.randint(FEAT_LENGTH_ - feat.shape[0] + 1)
                    l = random_shift
                    r = random_shift + feat.shape[0]
                    t[l:r] = feat
                # if len(feat) >= FEAT_LENGTH_, randomly cut the feat to data of length of FEAT_LENGTH_
                else:
                    random_shift = np.random.randint(feat.shape[0] - FEAT_LENGTH_ + 1)
                    l = random_shift
                    r = random_shift + FEAT_LENGTH_
                    t = feat[l:r]

                feat = t

            # Spec Augment
            if self.specaug:
                if True: # np.random.randint(3) <= 1:
                    feat = np.expand_dims(feat, axis=0)
                    feat_F = spec_augment.freq_mask(feat, 5 + 1, 1)
                    feat_T_F = spec_augment.time_mask(feat_F, 8 + 1, 1)
                    feat = np.squeeze(feat_T_F, axis=0)

            # For training
            if self.utt2id is not None:
                # Sparse softmax cross entropy label
                id = np.array(self.utt2id[utt], dtype=np.int32)
                # Dense softmax cross entropy label
                # id = np.zeros((self.num_classes,))
                # id[self.utt2id[utt]] = 1
            else:
                id = utt
            if self.utt2kwd is not None:
                kwd = self.utt2kwd[utt]
            else:
                kwd = 0

            yield feat, id
