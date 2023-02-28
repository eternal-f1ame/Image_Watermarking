import numpy as np

def test_random(T):
    cover_image = np.random.rand(100, 100, 3)*255
    cover_image = cover_image.astype(np.uint)
    K = 10
    W = np.random.rand(100, 100, 3)*255
    W = W.astype(np.uint)
    encrypted = T.Enc(cover_image, W, K)
    return np.array_equal(W,T.Dec(encrypted, cover_image, K))
