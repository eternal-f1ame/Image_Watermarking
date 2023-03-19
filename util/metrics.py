"""Metrics for evaluating the performance of a model."""
import numpy as np
from skimage.metrics import structural_similarity as ssim

def mse(_d, _i):
    """Mean Squared Error"""

    assert _d.shape == _i.shape, "Input images must have the same size"
    assert _d.ndim == _i.ndim, "Input images must have the same number of channels"

    return (np.square(_d - _i)).mean()

def psnr(_d, _i):
    """Peak Signal to Noise Ratio"""

    assert _d.shape == _i.shape, "Input images must have the same size"
    assert _d.ndim == _i.ndim, "Input images must have the same number of channels"

    return 10 * np.log10(1 / mse(_d, _i))


def ssim_(_d, _i):
    """Structural Similarity (SSIM) Index"""
    ssim_index =  ssim(_d, _i, channel_axis=2)
    return ssim_index


def rgb2ycbcr(img):
    """
    Convert an RGB image to YCbCr color space
    """

    _t = np.array([[0.299, 0.587, 0.114],
                  [-0.169, -0.331, 0.5],
                  [0.5, -0.419, -0.081]])
    t_inv = np.linalg.inv(_t)
    ycbcr = np.dot(img, _t.T)
    ycbcr[:,:,1:] += 128
    return ycbcr.astype(np.uint8)

def ber(_d, _i):
    """
    Bit Error Rate"""

    assert _d.shape == _i.shape, "Input images must have the same size"
    assert _d.ndim == _i.ndim, "Input images must have the same number of channels"

    if _d.ndim == 3 and _d.shape[-1] == 3:
        _d = rgb2ycbcr(_d)[:,:,0]
        _i = rgb2ycbcr(_i)[:,:,0]

    _d = _d.astype(np.uint8)
    _i = _i.astype(np.uint8)

    ber_val = np.sum(np.abs(_d - _i)) / (_d.size * 8)

    return ber_val

METRICS = {
    "MSE": mse,
    "PSNR": psnr,
    "SSIM": ssim_,
    "BER": ber
    }
