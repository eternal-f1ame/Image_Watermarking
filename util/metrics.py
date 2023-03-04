"""Metrics for evaluating the performance of a model."""
import numpy as np
from scipy import signal
from scipy import ndimage
import gauss

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

def ssim(img1, img2, cs_map=False):
    """Structural Similarity (SSIM) Index"""

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = gauss.fspecial_gauss(size, sigma)
    k_1 = 0.01
    k_2 = 0.03
    _l = 255 #bitdepth of image
    c_1 = (k_1*_l)**2
    c_2 = (k_2*_l)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + c_1)*(2*sigma12 + c_2))/((mu1_sq + mu2_sq + c_1)*
                    (sigma1_sq + sigma2_sq + c_2)), 
                (2.0*sigma12 + c_2)/(sigma1_sq + sigma2_sq + c_2))
    else:
        return ((2*mu1_mu2 + c_1)*(2*sigma12 + c_2))/((mu1_sq + mu2_sq + c_1)*
                    (sigma1_sq + sigma2_sq + c_2))

def msssim(img1, img2):
    """Multi-scale Structural Similarity (MSSSIM) Index"""

    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2))/4.0
    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)
    mssim = np.array([])
    mcs = np.array([])
    for _ in range(level):
        ssim_map, cs_map = ssim(im1, im2, cs_map=True)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.filters.convolve(im1, downsample_filter, 
                                                mode='reflect')
        filtered_im2 = ndimage.filters.convolve(im2, downsample_filter, 
                                                mode='reflect')
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    return (np.prod(mcs[0:level-1]**weight[0:level-1])*
                    (mssim[level-1]**weight[level-1]))

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
    "MSSIM": msssim,
    "BER": ber
    }
