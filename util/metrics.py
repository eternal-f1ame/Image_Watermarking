import numpy as np

def MSE(D, I):
    return (np.square(D - I)).mean()

METRICS = {"MSE": MSE,
        }
