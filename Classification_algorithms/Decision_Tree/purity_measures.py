import numpy as np


# There are many ways to compute a node purity, those are three chosen for this implementation

def entropy(counts):
    s = sum(counts)
    counts = counts / s
    return -np.sum(counts * np.log2(counts + 0.0001))


def gini(counts):
    s = sum(counts)
    counts = counts / s
    return 1 - np.sum(counts * counts)


def mean_err_rate(counts):
    counts = counts / sum(counts)
    return 1 - max(counts)
