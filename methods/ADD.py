"""Additive algorithm for Watermarking"""

import numpy as np
from ..util.watermarking import Watermarking
from ..util.test import encrypt_img
from ..util.test import eval_metrics

class ADD(Watermarking):
    """Class for watermarking for ADD algorithm."""
    def __init__(self):
        self.type = "ADD"

    def enc(self, _i, _w, _k):
        """Encrypts an image _i with a watermark _w and a key _k."""
        _w = cv2.resize(_w, (_i.shape[1], _i.shape[0]))
        alpha = _k
        i_res = _i + alpha*_w
        i_res = np.clip(i_res,0,255)
        return i_res

    def dec(self, _d, _i, _k):
        """Decrypts a watermark from an image _i with a key _k."""

        alpha = _k
        i_res = _i - _d
        i_res = i_res/alpha
        i_res = np.clip(i_res, 0, 255)
        return i_res

T = ADD()
I = cv2.imread()
W = cv2.imread()
K = 0.05
encrypted_image = T.enc(I, W, K)
decrypted_image = T.dec(I, encrypted_image, K)
