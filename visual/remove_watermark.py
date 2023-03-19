import cv2
import os
import numpy as np

# Step 1
def estimate(path_name):
    gradient_x = []
    gradient_y = []
    for img_name in os.listdir(path_name):
        img = cv2.imread(os.path.join(path_name, img_name))
        gradient_x.append(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3))
        gradient_y.append(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3))
    
    res_x = np.median(np.array(gradient_x), axis=0)
    res_y = np.median(np.array(gradient_y), axis=0)
    return res_x, res_y

def crop_watermark(Dx, Dy):
    D_mod = np.sqrt(np.square(Dx) + np.square(Dy))
    D_mod = D_mod.astype(float)
    D_mod = (D_mod - np.min(D_mod))/(np.max(D_mod)-np.min(D_mod))
    D_mod[D_mod >= 0.2] =1
    D_mod[D_mod < 0.2] = 0
    return D_mod

def remove_watermark():
    for i in range(10):
        path_name = "outputs/"
        Dx, Dy = estimate(path_name)
        wmk = crop_watermark(Dx, Dy)
        wmk = (255*wmk).astype(np.uint8)
        cv2.imshow("wmk", wmk)
        cv2.waitKey(0)
        wmk_orig = cv2.imread("Watermark.png")
        wmk_orig = cv2.resize(wmk_orig, (300,300))
        for img_name in os.listdir("outputs/"):
            img = cv2.imread(os.path.join(path_name, img_name))

            #cv2.imshow("watermark",cv2.subtract(img,(0.2*255*(wmk)).astype(np.uint8)))
            #cv2.imshow(img_name, cv2.addWeighted(img, 1, (255*wmk).astype(np.uint8), 0.3,0))
            cv2.imwrite(os.path.join(path_name,img_name), (cv2.addWeighted(img, 1, wmk, 0.06,0)))
            #cv2.waitKey(0)

def preview():
    path_name = "removal_results/"
    for img_name in os.listdir("removal_results/"):
        img = cv2.imread(os.path.join(path_name, img_name))
        #cv2.imshow("watermark",cv2.subtract(img,(0.2*255*(wmk)).astype(np.uint8)))
        cv2.imshow(img_name, cv2.addWeighted(img, 1.25, np.ones(img.shape).astype(np.uint8), 0, 0))
        cv2.waitKey(0)

remove_watermark()
preview()



