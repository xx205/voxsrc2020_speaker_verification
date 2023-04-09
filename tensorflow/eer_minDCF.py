import sys
import argparse

import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def read_score_file(score_file):
    pair_score = {}
    for line in open(score_file, 'r'):
        utt1, utt2, score = line.strip().split()
        score = float(score)
        pair_score[(utt1, utt2)] = score
    return pair_score


def read_trial_file(trial_file):
    pair_label = {}
    for line in open(trial_file, 'r'):
        label, utt1, utt2 = line.strip().split()
        label = int(label)
        pair_label[(utt1, utt2)] = label
    return pair_label


def compute_eer_and_min_dcf(y, y_pred, c_miss, c_fa, p_target):
    fprs, tprs, thresholds = roc_curve(y, y_pred, pos_label=1)
    fnrs = 1.0 - tprs

    eer_threshold = thresholds[np.nanargmin(np.absolute((fnrs - fprs)))]
    eer = fprs[np.nanargmin(np.absolute((fnrs - fprs)))]
    # eer = fnrs[np.nanargmin(np.absolute((fnrs - fprs)))]

    # eer = brentq(lambda x : 1. - x - interp1d(fprs, tprs)(x), 0., 1.)
    # eer_threshold = interp1d(fprs, thresholds)(eer)

    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    
    return eer, eer_threshold, min_dcf, min_c_det_threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c-miss', type=float, dest = "c_miss", default = 1,
                        help='Cost of a missed detection.  This is usually not changed.')
    parser.add_argument('--c-fa', type=float, dest = "c_fa", default = 1,
                        help='Cost of a spurious detection.  This is usually not changed.')
    parser.add_argument('--p-target', type=float, dest = "p_target", default = 0.01,
                        help='The prior probability of the target speaker in a trial.')
    parser.add_argument("--trial", type=str,
                        help="the trial file")
    parser.add_argument("--score", type=str,
                        help="the score file")
    args = parser.parse_args()

    pair_label = read_trial_file(args.trial)
    pair_score = read_score_file(args.score)
    c_miss = args.c_miss
    c_fa = args.c_fa
    p_target = args.p_target
    
    y, y_pred = [], []
    for pair in pair_label.keys():
        y.append(pair_label[pair])
        y_pred.append(pair_score[pair])

    eer, eer_threshold, min_dcf, min_c_det_threshold = compute_eer_and_min_dcf(y, y_pred, c_miss, c_fa, p_target)
    print("EER is {:.4f}%, at threshold: {:.4f}".format(eer * 100, eer_threshold))
    print("minDCF is {:.4f}, at threshold: {:.4f} (p-target={}, c-miss={}, c-fa={})".format(min_dcf, min_c_det_threshold, p_target, c_miss, c_fa))


if __name__ == '__main__':
    main()
