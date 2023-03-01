import numpy as np
import cv2

def test_random(T):
    cover_image = np.random.rand(100, 100, 3)*255
    cover_image = cover_image.astype(np.uint)
    K = 10
    W = np.random.rand(100, 100, 3)*255
    W = W.astype(np.uint)
    encrypted = T.Enc(cover_image, W, K)
    return np.array_equal(W,T.Dec(encrypted, cover_image, K))

def encrypt_img(T, img_path, K=10):
   cover_image = cv2.imread(img_path)
   cover_image = cover_image.astype(np.uint8)
   W = np.random.rand(cover_image.shape[0],cover_image.shape[1],cover_image.shape[2])*2
   W = W.astype(np.uint8)
   encrypted = T.Enc(cover_image, W, K)
   cv2.imshow("enc",encrypted)
   cv2.waitKey(0)
   return T.Enc(cover_image, W, K)
   
