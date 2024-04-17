#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:21:30 2021

@author: Markus Meister
"""
import torch
from scipy.stats import wasserstein_distance


def compute_probs(data, n=10, lo=0, hi=0):
    h = torch.histc(data.float(), n, lo, hi)
    p = h / data.shape[0]
    return p


def kl_divergence(p, q):
    return (p * (p / q).log()).sum()


def js_divergence(p, q):
    m = (1.0 / 2.0) * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def ws_divergence(p, q):
    dws = wasserstein_distance(p, q)
    return torch.tensor(dws)


def compute_div_fun(div_fun, sample_1, sample_2, n_bins=10):
    if not isinstance(sample_1, torch.Tensor):
        sample_1 = torch.tensor(sample_1)
    if not isinstance(sample_2, torch.Tensor):
        sample_2 = torch.tensor(sample_2)

    # limits
    lo = min(sample_1.min(), sample_2.min())
    hi = max(sample_1.max(), sample_2.max())

    # histograms
    p = compute_probs(sample_1, n_bins, lo, hi)
    q = compute_probs(sample_2, n_bins, lo, hi)

    # list_of_tuples = support_intersection(p,q)
    # p, q = get_probs(list_of_tuples)

    # intersections
    zp = p != 0
    zq = q != 0
    zd = zp & zq
    p = p[zd]
    q = q[zd]

    # final divergence
    return div_fun(p, q)


def compute_kl_divergence(sample_1, sample_2, n_bins=10):
    """
    Computes the KL Divergence using the support
    intersection between two different samples
    """
    return compute_div_fun(kl_divergence, sample_1, sample_2, n_bins)


def compute_js_divergence(sample_1, sample_2, n_bins=10):
    """
    Computes the JS Divergence using the support
    intersection between two different samples
    """
    return compute_div_fun(js_divergence, sample_1, sample_2, n_bins)


def compute_ws_divergence(sample_1, sample_2, n_bins=10):
    """
    Computes the JS Divergence using the support
    intersection between two different samples
    """
    return compute_div_fun(ws_divergence, sample_1, sample_2, n_bins)


def compute_ws_distance(sample_1, sample_2):
    return wasserstein_distance(sample_1, sample_2)
