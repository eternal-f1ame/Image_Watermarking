from util.watermarking import Watermarking
from util.test import test_random
from util.test import encrypt_img
from util.test import eval_metrics
import numpy as np

class LSB(Watermarking):
    def __init__(self):
        pass

    def Enc(self, I, W, K):
        I_flat = I.flatten()
        W_bits = W + bytes("$$$",encoding='utf8')
        cnt = 0
        for b in W_bits:
            for i in range(8):
                bit_val = (b >> i) & 1
                if bit_val == 1:
                    I_flat[cnt] |= 1
                else:
                    I_flat[cnt] &= 0
                cnt += 1
        I_res = np.resize(I_flat, I.shape)
        return I_res

    def Dec(self, D, I, K):
        D_flat = D.flatten()
        cnt = 0
        recovered = bytearray()
        for i in range(len(D_flat)):
            current_byte = 0
            for j in range(8):
                current_byte |= ((D_flat[cnt] & 1)<<j)
                cnt+=1
                if cnt >= len(D_flat):
                    break
            if cnt >= len(D_flat):
                break
            recovered += int(current_byte).to_bytes(1, 'big')
            if recovered[-3:] == b'$$$':
                break
        return bytes(recovered[:-3])

T = LSB()
W = b"1100"
D = (T.Enc(np.eye(10,10,3).astype(np.uint8),W,10))
print(eval_metrics(T, "../test_images/img_1.jpeg", b"110000"))
encrypt_img(T, "../test_images/img_1.jpeg", b"1111101010010101010010101100100101000101010101001010")
print(test_random(T))
