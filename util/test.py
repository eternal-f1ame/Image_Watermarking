"""Test the encryption and decryption of an image"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from util.metrics import METRICS

def test_random(_t, w_size=4):
    """Test the encryption and decryption of a random image"""

    cover_image = np.random.rand(100, 100, 3)*255
    cover_image = cover_image.astype(np.uint8)
    _k = 10
    _w = os.urandom(w_size)
    if _t.type in ['DCT', 'DWT']:
        _w = str(_w)
    encrypted = _t.enc(cover_image, _w, _k)
    if _t.type in ['DCT', 'DWT']:
        cover_image = _t.add_padd(cover_image)
    return np.array_equal(_w,_t.dec(encrypted, cover_image, _k))

def encrypt_img(_t, img_path, _w, _k=10):
    """Encrypt an image and return the encrypted image"""

    cover_image = cv2.imread(img_path)
    cover_image = cover_image.astype(np.uint8)
    encrypted = _t.enc(cover_image, _w, _k)
    if _t.type in ['DCT', 'DWT']:
        cover_image = _t.add_padd(cover_image)
    plt.figure("Comparative",(20,10))
    plt.subplot(1,2,1)
    plt.imshow(encrypted)
    plt.title("Encrypted Image")
    plt.subplot(1,2,2)
    plt.imshow(cover_image)
    plt.title("Cover Image")
    plt.show()
    return _t.enc(cover_image, _w, _k)

def eval_metrics(_t, img_path, _w, _k = 10):
    """Evaluate the metrics of an image"""

    cover_img = cv2.imread(img_path)
    cover_img = cover_img.astype(np.uint8)
    _d = _t.enc(cover_img, _w, _k)

    if _t.type in ['DCT', 'DWT']:
        cover_img = _t.add_padd(cover_img)

    metrics = {}
    for metric in METRICS.keys():
        metrics[metric] = METRICS[metric](_d, cover_img)
    return metrics

def eval_metrics_cv(_t, img, _w, _k = 10):
    """Evaluate the metrics of an image"""

    cover_img = img
    cover_img = cover_img.astype(np.uint8)
    _d = _t.enc(cover_img, _w, _k)
    metrics = {}
    for metric in METRICS.keys():
        metrics[metric] = METRICS[metric](_d, cover_img)
    return metrics
