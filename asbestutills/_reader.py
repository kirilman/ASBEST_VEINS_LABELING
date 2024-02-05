import numpy as np

def read_segmentation_labels(p):
    with open(p, 'r') as f:
        lines = f.readlines()
        return [np.fromstring(line, sep=' ').tolist() for line in lines]
