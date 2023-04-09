#!/usr/bin/env python3

# original paper: https://arxiv.org/pdf/1904.08779.pdf
# codes come from https://github.com/zcaceres/spec_augment
# this modified version accepts numpy array instead of pytorch tensor

from collections import namedtuple
import random


# def freq_mask(spec, F=30, num_masks=1, replace_with_zero=False):
def time_mask(spec, F=30, num_masks=1, replace_with_zero=True):
    # cloned = spec.clone()
    cloned = spec.copy()
    num_mel_channels = cloned.shape[1]

    for i in range(0, num_masks):
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)

        # avoids randrange error if values are equal and range is empty
        if (f_zero == f_zero + f): return cloned

        mask_end = random.randrange(f_zero, f_zero + f)
        if (replace_with_zero): cloned[0][f_zero:mask_end] = 0
        else: cloned[0][f_zero:mask_end] = cloned.mean()

    return cloned


# def time_mask(spec, T=40, num_masks=1, replace_with_zero=False):
def freq_mask(spec, T=40, num_masks=1, replace_with_zero=True):
    # cloned = spec.clone()
    cloned = spec.copy()
    len_spectro = cloned.shape[2]

    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t): return cloned

        mask_end = random.randrange(t_zero, t_zero + t)
        if (replace_with_zero): cloned[0][:,t_zero:mask_end] = 0
        else: cloned[0][:,t_zero:mask_end] = cloned.mean()
    return cloned
