import numpy as np


def compute_hinge(x):
    return np.maximum(0, x)


def scale_back_matrix_entry(entry, max_val, min_val, digits=4):
    return format((max_val - min_val) * entry + min_val, f'.{digits}g')