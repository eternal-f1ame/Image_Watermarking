"""Test the encryption and decryption of an image"""

import os
import numpy as np
import cv2
from util.metrics import METRICS

def test_random(_t, w_size=4):
    """Test the encryption and decryption of a random image"""

    cover_image = np.random.rand(100, 100, 3)*255
    cover_image = cover_image.astype(np.uint8)
    _k = 10
    _w = os.urandom(w_size)
    encrypted = _t.Enc(cover_image, _w, _k)
    return np.array_equal(_w,_t.Dec(encrypted, cover_image, _k))
   
def encrypt_img(_t, img_path, _w, _k=10):
    """Encrypt an image and return the encrypted image"""

    cover_image = cv2.imread(img_path)
    cover_image = cover_image.astype(np.uint8)
    encrypted = _t.Enc(cover_image, _w, _k)
    cv2.imshow("enc",encrypted)
    cv2.waitKey(0)
    return _t.Enc(cover_image, _w, _k)
   
def eval_metrics(_t, img_path, _w, _k = 10):
    """Evaluate the metrics of an image"""

    cover_img = cv2.imread(img_path)
    cover_img = cover_img.astype(np.uint8)
    D = _t.Enc(cover_img, _w, _k)
    metrics = {}
    for metric in METRICS.items():
        metric = metric(D, cover_img)
    return metrics

def eval_metrics_cv(_t, img, _w, _k = 10):
    """Evaluate the metrics of an image"""

    cover_img = img
    cover_img = cover_img.astype(np.uint8)
    D = _t.Enc(cover_img, _w, _k)
    metrics = {}
    for metric in METRICS.items():
        metric = metric(D, cover_img)
    return metrics
