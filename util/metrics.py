"""Metrics for evaluating the performance of a model."""
import numpy as np

def mse(_d, _i):
    """Mean Squared Error"""
    return (np.square(_d - _i)).mean()

METRICS = {"MSE": mse,
        }
