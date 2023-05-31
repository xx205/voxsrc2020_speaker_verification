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


import argparse
import pickle

import numpy as np
import kaldi_io
# from collections import OrderedDict

def l2norm(x, axis=0, keepdims=True):
    x_norm = np.linalg.norm(x, axis=axis, keepdims=keepdims)
    return x / x_norm


def read_xvector(xvector_ark):
    # xvectors = OrderedDict([])
    xvectors = {}
    for utt, vec in kaldi_io.read_vec_flt_ark(xvector_ark):
        xvectors[utt] = l2norm(vec, axis=0)
    return xvectors


def read_spk2utt(spk2utt_file):
    spk2utt = {}
    for line in open(spk2utt_file, 'r'):
        spk_utt = line.strip().split()
        spk, utt = spk_utt[0], spk_utt[1:]
        spk2utt[spk] = utt
    return spk2utt


def read_speaker_xvector(xvectors, spk2utt):
    utt_to_spk = {}
    for spk, utts in spk2utt.items():
        for utt in utts:
            utt_to_spk[utt] = spk

    # speaker_xvectors = OrderedDict([])
    speaker_xvectors = {}
    for utt, vec in xvectors.items():
        if utt in utt_to_spk:
            spk = utt_to_spk[utt]
            if spk not in speaker_xvectors:
                speaker_xvectors[spk] = [vec]
            else:
                speaker_xvectors[spk].append(vec)

    for spk, vecs in speaker_xvectors.items():
        matrix = np.array(vecs)
        matrix = l2norm(matrix, axis=1)
        vec = np.mean(matrix, axis=0)

        speaker_xvectors[spk] = vec
    return speaker_xvectors


def get_cohort_xvector(cohort_ark, cohort_spk2utt):
    xvectors = read_xvector(cohort_ark)
    spk2utt = read_spk2utt(cohort_spk2utt)
    speaker_xvectors = read_speaker_xvector(xvectors, spk2utt)
    return speaker_xvectors


def get_projection_weight(weight_matrix_pkl):
    weight_matrix = pickle.load(open(weight_matrix_pkl, 'rb'))
    speaker_xvectors= {i:l2norm(weight_matrix[i]) for i in range(len(weight_matrix))}
    return speaker_xvectors


def get_cohort_mean_std(trial_xvectors, cohort_xvectors, topk=400):
    utt = list(trial_xvectors.keys())
    trial_matrix = np.array(list(trial_xvectors.values()))
    cohort_matrix = np.array(list(cohort_xvectors.values()))
    # cohort_scores = np.matmul(trial_matrix, np.transpose(cohort_matrix))
    # topk_cohort_scores = (-1 * np.sort(-cohort_scores, axis=1))[:, :topk]
    # mean = np.mean(topk_cohort_scores, axis=1)
    # std = np.std(topk_cohort_scores, axis=1)

    # utt_to_mean = {}
    # utt_to_std = {}
    # for u, m, s in zip(utt, mean, std):
    #     utt_to_mean[u] = m
    #     utt_to_std[u] = s
    utt_to_mean, utt_to_std = {}, {}
    cohort_matrix_T = np.transpose(cohort_matrix)
    sub_size = 1024
    for i in range(0, len(trial_matrix), sub_size):
        i_end = min(i + sub_size, len(trial_matrix))
        sub_trial_matrix = trial_matrix[i:i_end, :]
        cohort_scores = np.matmul(sub_trial_matrix, cohort_matrix_T)
        topk_cohort_scores = (-1 * np.sort(-cohort_scores, axis=1))[:, :topk]
        mean = np.mean(topk_cohort_scores, axis=1)
        std = np.std(topk_cohort_scores, axis=1)

        for j in range(i, i_end):
            utt_to_mean[utt[j]], utt_to_std[utt[j]] = mean[j % 1024], std[j % 1024]
    return utt_to_mean, utt_to_std


def get_cosine_score(trial_xvectors, trial):
    scores = []
    for line in open(trial, 'r'):
        utt1, utt2 = line.strip().split()[-2:]
        score = np.dot(trial_xvectors[utt1],
                       trial_xvectors[utt2])
        scores.append((utt1, utt2, score))
    return scores


def get_asnorm1_score(utt_to_mean, utt_to_std, scores):
    asnorm1_scores = []
    for (utt1, utt2, score) in scores:
        asnorm1_scores.append(
            (utt1,
             utt2,
             0.5 * ((score - utt_to_mean[utt1]) / utt_to_std[utt1] +
                    (score - utt_to_mean[utt2]) / utt_to_std[utt2])))
    return asnorm1_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_ark", type=str,
                        help="the ark file of test xvectors")    
    parser.add_argument("--test_spk2utt", type=str, default=None,
                        help="the spk2utt file of test speakers")
    parser.add_argument("--trial", type=str,
                        help="the trial file")
    parser.add_argument("--cosine_score", type=str,
                        help="cosine score file")
    parser.add_argument("--cohort_ark", type=str, default=None, required=False,
                        help="the ark file of cohort xvectors")
    parser.add_argument("--cohort_spk2utt", type=str, default=None, required=False,
                        help="the spk2utt file of cohort xvectors")
    parser.add_argument("--weight_matrix", type=str, default=None, required=False,
                        help="the projection weight matrix file")
    parser.add_argument("--snorm_score", type=str, default=None, required=False,
                        help="snorm score file")
    
    args = parser.parse_args()

    test_xvectors = read_xvector(args.test_ark)

    if args.test_spk2utt is not None:
        test_spk2utt = read_spk2utt(args.test_spk2utt)
        test_speaker_xvectors = read_speaker_xvector(test_xvectors, test_spk2utt)
        test_xvectors.update(test_speaker_xvectors)

    cosine_score = get_cosine_score(test_xvectors, args.trial)

    with open(args.cosine_score, 'w') as f:
        for (utt1, utt2, score) in cosine_score:
            print(utt1, utt2, score, file=f)

    if args.snorm_score is not None:
        if args.cohort_ark is not None and args.cohort_spk2utt is not None:
            cohort_speaker_xvectors = get_cohort_xvector(args.cohort_ark, args.cohort_spk2utt)
        elif args.weight_matrix is not None:
            cohort_speaker_xvectors = get_projection_weight(args.weight_matrix)
        else:
            raise ValueError("Can not compute snorm scores: no cohort vectors provided")

        utt_to_mean, utt_to_std = get_cohort_mean_std(test_xvectors, cohort_speaker_xvectors)

        asnorm1_score = get_asnorm1_score(utt_to_mean, utt_to_std, cosine_score)
        
        with open(args.snorm_score, 'w') as f:
            for (utt1, utt2, score) in asnorm1_score:
                print(utt1, utt2, score, file=f)
