"""Metrics for evaluating the performance of a model."""
import numpy as np
from scipy import signal

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

def mpsnr(_d, _i, mask):
    """
    Masking-based Peak Signal to Noise Ratio
    """

    assert _d.shape == _i.shape, "Input images must have the same size"
    assert _d.ndim == _i.ndim, "Input images must have the same number of channels"

    d_masked = _d[mask]
    i_masked = _i[mask]

    mse_masked = mse(d_masked, i_masked)
    max_pixel_value = np.iinfo(_d.dtype).max
    psnr_masked = 10 * np.log10(max_pixel_value**2 / mse_masked)
    return psnr_masked

def mssim(_d, _i, k_1=0.01, k_2=0.03, sigma=1.5, win_size=11):
    """
    Mean Structural Similarity (MSSIM) Index
    """

    assert _d.shape == _i.shape, "Input images must have the same size"
    assert _d.ndim == _i.ndim, "Input images must have the same number of channels"

    if _d.ndim == 3 and _d.shape[-1] == 3:
        _d = rgb2ycbcr(_d)[:,:,0]
        _i = rgb2ycbcr(_i)[:,:,0]

    _d = _d.astype(np.float64) / np.max(_d)
    _i = _i.astype(np.float64) / np.max(_i)

    mu_d = signal.convolve2d(_d, gaussian_kernel(sigma, win_size), mode='same')
    mu_i = signal.convolve2d(_i, gaussian_kernel(sigma, win_size), mode='same')
    sigma_d = signal.convolve2d(_d**2, gaussian_kernel(sigma, win_size), mode='same') - mu_d**2
    sigma_i = signal.convolve2d(_i**2, gaussian_kernel(sigma, win_size), mode='same') - mu_i**2
    sigma_di = signal.convolve2d(_d*_i, gaussian_kernel(sigma, win_size), mode='same') - mu_d*mu_i

    c_1 = (k_1*np.max(_d))**2
    c_2 = (k_2*np.max(_d))**2
    num = (2*mu_d*mu_i + c_1)*(2*sigma_di + c_2)
    den = (mu_d**2 + mu_i**2 + c_1)*(sigma_d + sigma_i + c_2)

    mssim_val = np.mean(num / den)
    return mssim_val

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

def gaussian_kernel(sigma, win_size):
    """
    Compute a 2D Gaussian kernel
    """
    assert win_size % 2 == 1, "Kernel size must be odd"
    k = (win_size - 1) // 2
    _x, _y = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))
    kernel = np.exp(-(_x**2 + _y**2) / (2*sigma**2))
    return kernel / np.sum(kernel)

METRICS = {
    "MSE": mse,
    "PSNR": psnr,
    "MPSNR": mpsnr,
    "MSSIM": mssim,
    "BER": ber
    }
