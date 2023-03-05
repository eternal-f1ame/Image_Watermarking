"""LSB algorithm for watermarking."""

import numpy as np
from util.watermarking import Watermarking
from util.test import test_random
from util.test import encrypt_img
from util.test import eval_metrics

class LSB(Watermarking):
    """Abstract class for watermarking for LSB algorithm."""
    def __init__(self):
        pass

    def enc(self, _i, _w, _k):
        """Encrypts an image _i with a watermark _w and a key _k."""
        i_flat = _i.flatten()
        w_bits = _w + bytes("$$$",encoding='utf8')
        cnt = 0
        for bit in w_bits:
            for i in range(8):
                bit_val = (bit >> i) & 1
                if bit_val == 1:
                    i_flat[cnt] |= 1
                else:
                    i_flat[cnt] &= 0
                cnt += 1
        i_res = np.resize(i_flat, _i.shape)
        return i_res

    def dec(self, _d, _i, _k):
        """Decrypts an image _d with a key _k."""
        d_flat = _d.flatten()
        cnt = 0
        recovered = bytearray()
        for _ in range(len(d_flat)):
            current_byte = 0
            for j in range(8):
                current_byte |= ((d_flat[cnt] & 1)<<j)
                cnt+=1
                if cnt >= len(d_flat):
                    break
            if cnt >= len(d_flat):
                break
            recovered += int(current_byte).to_bytes(1, 'big')
            if recovered[-3:] == b'$$$':
                break
        return bytes(recovered[:-3])

T = LSB()
W = b"1100"
_d = (T.enc(np.eye(10,10,3).astype(np.uint8),W,10))
print(eval_metrics(T, "../test_images/img_1.jpeg", b"110000"))
encrypt_img(T, "../test_images/img_1.jpeg", b"1111101010010101010010101100100101000101010101001010")
print(test_random(T))
