import torch


def std_threshold(metric, thr=4.0):
    n_data = len(metric)
    outs = torch.ones(n_data)
    while sum(outs) > 0.5*n_data:
        outs = metric > thr*metric.std()
        thr += .1
    return outs
