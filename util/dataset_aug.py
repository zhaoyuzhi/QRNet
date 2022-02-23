import numpy as np
from PIL import Image, ImageEnhance
import cv2
import random
import skimage
import colour_demosaicing
import math
from scipy.stats import tukeylambda
import torch
import torch.nn as nn
import torch.nn.functional as F

def sobel(img):
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0) #x方向的梯度
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1) #y方向的梯度
    sobelX = np.uint8(np.absolute(sobelX)) #x方向梯度的绝对值
    sobelY = np.uint8(np.absolute(sobelY)) #y方向梯度的绝对值
    sobelCombined = cv2.bitwise_or(sobelX,sobelY)
    return sobelCombined

def add_noise_raw(raw, K = 1, sigma = 10):

    # Assume now already subtracted black level
    raw = np.clip((raw * 255), 0, 255)

    # Shot noise(Possion noise)
    #K_g, K_r, K_b = K, K, K
    #raw[::2, ::2]   = np.random.poisson(raw[::2, ::2] * K_r) / (K_r)    # R
    #raw[1::2, ::2]  = np.random.poisson(raw[1::2, ::2] * K_g) / (K_g)   # G
    #raw[::2, 1::2]  = np.random.poisson(raw[::2, 1::2] * K_g) / (K_g)   # G
    #raw[1::2, 1::2] = np.random.poisson(raw[1::2, 1::2] * K_b) / (K_b)  # B
    raw = np.random.poisson(raw * K) / K
    raw = np.clip(raw, 0, 255)

    # Read noise(addtion noise) 
    gaussian_noise = np.random.normal(loc = 0.0, scale = sigma, size = raw.shape)
    raw = raw + gaussian_noise

    raw = np.clip(raw, 0, 255)
    raw = np.clip(raw / 255, 0, 1)

    return raw

def add_noise_unprocessing_long(x, K = 1, sigma = 10):

    # Raw container
    raw = np.zeros(x.shape[:2], dtype = np.float32)

    x = np.power(x, 2.2)                    # Inverse gamma

    # Resampling
    raw[::2, ::2] = x[::2, ::2, 0]          # R
    raw[1::2, ::2] = x[1::2, ::2, 1]        # G
    raw[::2, 1::2] = x[::2, 1::2, 1]        # G
    raw[1::2, 1::2] = x[1::2, 1::2, 2]      # B

    # Inverse AWB, randomly choosing AWB parameters
    # from red [1.9, 2.4], blue [1.5, 1.9]
    awb_b, awb_r = 0.5 * random.random() + 1.9, 0.4 * random.random() + 1.5
    raw[::2, ::2] /= awb_r  # awb_r
    raw[1::2, 1::2] /= awb_b  # awb_b

    # Assume now already subtracted black level
    raw = np.clip((raw * 255), 0, 255)

    # Shot noise(Possion noise)
    K_g, K_r, K_b = K, K, K
    raw[::2, ::2]   = np.random.poisson(raw[::2, ::2] * K_r) / (K_r)    # R
    raw[1::2, ::2]  = np.random.poisson(raw[1::2, ::2] * K_g) / (K_g)   # G
    raw[::2, 1::2]  = np.random.poisson(raw[::2, 1::2] * K_g) / (K_g)   # G
    raw[1::2, 1::2] = np.random.poisson(raw[1::2, 1::2] * K_b) / (K_b)  # B
    raw = np.clip(raw, 0, 255)

    # Read noise(addtion noise) 
    gaussian_noise = np.random.normal(loc = 0.0, scale = sigma, size = raw.shape)
    raw = raw + gaussian_noise
    raw = np.clip(raw, 0, 255)

    # AWB and quantization noise
    raw[::2, ::2] *= awb_r  # awb_r
    raw[1::2, 1::2] *= awb_b   # awb_b
    raw = np.clip(raw, 0, 255).astype(np.uint8)

    demosaicked_rgb = colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(raw, 'RGGB')
    demosaicked_rgb = np.clip(demosaicked_rgb / 255, 0, 1)
    x = np.power(demosaicked_rgb, 1 / 2.2)

    return x

def add_noise_unprocessing_short(x, K = 1, sigma = 10):

    # Raw container
    raw = np.zeros(x.shape[:2], dtype = np.float32)

    x = np.power(x, 2.2)                    # Inverse gamma

    # Resampling
    raw[::2, ::2] = x[::2, ::2, 0]          # R
    raw[1::2, ::2] = x[1::2, ::2, 1]        # G
    raw[::2, 1::2] = x[::2, 1::2, 1]        # G
    raw[1::2, 1::2] = x[1::2, 1::2, 2]      # B

    # Inverse AWB, randomly choosing AWB parameters
    # from red [1.9, 2.4], blue [1.5, 1.9]
    awb_b, awb_r = 0.5 * random.random() + 1.9, 0.4 * random.random() + 1.5
    raw[::2, ::2] /= awb_r  # awb_r
    raw[1::2, 1::2] /= awb_b  # awb_b

    # Assume now already subtracted black level
    raw = np.clip((raw * 255), 0, 255)

    # Amplification ratio
    raw = raw / 4

    # Shot noise(Possion noise)
    K_g, K_r, K_b = K, K, K
    raw[::2, ::2]   = np.random.poisson(raw[::2, ::2] * K_r) / (K_r)    # R
    raw[1::2, ::2]  = np.random.poisson(raw[1::2, ::2] * K_g) / (K_g)   # G
    raw[::2, 1::2]  = np.random.poisson(raw[::2, 1::2] * K_g) / (K_g)   # G
    raw[1::2, 1::2] = np.random.poisson(raw[1::2, 1::2] * K_b) / (K_b)  # B
    raw = np.clip(raw, 0, 255 / 4)

    # Read noise(addtion noise)
    gaussian_noise = np.random.normal(loc = 0.0, scale = sigma, size = raw.shape)
    raw = raw + gaussian_noise
    raw = np.clip(raw, 0, 255 / 4)

    # Amplification ratio
    raw = raw * 4

    # AWB and quantization noise
    raw[::2, ::2] *= awb_r  # awb_r
    raw[1::2, 1::2] *= awb_b   # awb_b
    raw = np.clip(raw, 0, 255).astype(np.uint8)

    demosaicked_rgb = colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(raw, 'RGGB')
    demosaicked_rgb = np.clip(demosaicked_rgb / 255, 0, 1)
    x = np.power(demosaicked_rgb, 1 / 2.2)

    return x

def apply_darken(img_array, enhance_factor, gamma_factor):
    """
    img_array: HWC, pixel should be in [0,1]
    """
    img_array = (img_array * 255.).astype(np.uint8)
    pil_img = Image.fromarray(img_array) 
    brightness_enhance = ImageEnhance.Brightness(pil_img)
    pil_img = brightness_enhance.enhance(enhance_factor)
    
    img_array = np.asarray(pil_img)
    img_float = img_array.astype(np.float32) / 255.0
    img_float = img_float ** gamma_factor
    # img_array = (img_float * 255.0).astype(np.uint8)

    return img_float

def darken(images, dark_prob = 0.5):
    """
    images: numpy array, pixel should be in [0, 1]
    """
    p = random.random()
    if p < dark_prob:
        enhance_factor = random.choice([0.6, 0.7, 0.8, 0.9])
        gamma_factor = 1 / random.choice([0.6, 0.7, 0.75, 0.8, 0.9])
        if len(images.shape) == 4:
            temp = []
            for i in range(images.shape[0]):
                image = apply_darken(images[i], enhance_factor, gamma_factor)
                temp.append(image)
            images = np.array(temp)
        elif len(images.shape) == 3:
            images = apply_darken(images, enhance_factor, gamma_factor)
        else:
            raise ValueError('Image shape %s is not supported.' % str(images.shape))

        return images
    else:
        return images

def color_distortion(img, prob = 0.5):
    """
    img: HWC(RGB format) or HW, value in [0, 1]
    """
    p = random.uniform(0.0, 1.0)
    if p < prob:
        img = img.astype(np.float32)
        if len(img.shape) == 2:
            a = random.uniform(0.3, 0.6)
            b = random.uniform(0.001, 0.01)
            img = a * img + b
        elif len(img.shape) == 3:
            a = random.uniform(0.6, 0.9)
            b = random.uniform(0.001, 0.01)
            _, _, C = img.shape
            for i in range(C):
                img[:, :, i] = a * img[:, :, i] + b
        else:
            raise ValueError("img shape %s is not supported." % (img.shape))
        
        return np.clip(img, 0.0, 1.0)
    else:
        return img

# ----------------------------------------------------------------
# def eval_K(iso):
#     iso_log = math.log10(iso)
#     K_g = iso_log ** 2.0 * 0.60948693 + iso_log * -5.27886693 + 11.51749064
#     K_r = iso_log ** 2.0 * 0.51866798 + iso_log * -4.55091212 + 10.04344718
#     K_b = iso_log ** 2.0 * 0.6151894  + iso_log * -5.38666882 + 11.86832215
#     return iso_log, K_g, K_r, K_b
# 
# print('iso_log', 'K_g', 'K_r', 'K_b:')
# 
# iso_log, K_g, K_r, K_b = eval_K(3200)
# print(iso_log, K_g, K_r, K_b)
# 
# iso_log, K_g, K_r, K_b = eval_K(6400)
# print(iso_log, K_g, K_r, K_b)
# 
# iso_log, K_g, K_r, K_b = eval_K(12800)
# print(iso_log, K_g, K_r, K_b)
# 
# print('read noise:')
# 
# gaussian_noise = np.random.normal(loc = 0.0, scale = 10.0, size = [2,2])
# print(gaussian_noise)
# 
# sigma = math.exp(math.log(K_b) * (-0.888) + 0.116)
# tl_addition = tukeylambda.rvs(0.09, size = [2,2], loc = 0.0, scale = sigma)
# print(sigma)
# print(tl_addition)
# ----------------------------------------------------------------

if __name__ == "__main__":

    #SBATCH --nodelist=HK-IDC2-10-1-75-60

    def noisy_long_img_gen(img, K = 0.1, sigma = 10):
        img_ori = img.copy()
        img = img / 255.0
        # add noise
        img = add_noise_unprocessing_long(img, K = K, sigma = sigma)
        #img = np.clip(img, 0, 0.25)
        # recover noisy image
        img = (img * 255.0).astype(np.uint8)
        # show
        img_concat = np.concatenate((img_ori, img), 1)
        return img_concat
        
    def noisy_short_img_gen(img, K = 0.1, sigma = 10):
        img_ori = img.copy()
        img = img / 255.0
        # add noise
        img = add_noise_unprocessing_short(img, K = K, sigma = sigma)
        #img = np.clip(img, 0, 0.25)
        # recover noisy image
        img = (img * 255.0).astype(np.uint8)
        # show
        img_concat = np.concatenate((img_ori, img), 1)
        return img_concat
    
    def noisy_raw_gen(img, K = 0.1, sigma = 10):
        img_ori = img.copy()
        img = img / 255.0
        # add noise
        img = add_noise_raw(img, K = K, sigma = sigma)
        # recover noisy image
        img = (img * 255.0).astype(np.uint8)
        # show
        img_concat = np.concatenate((img_ori, img), 1)
        return img_concat
    
    def darken_img_gen(img):
        img_ori = img.copy()
        img = img / 255.0
        # add darken
        img = darken(img, dark_prob = 1)
        # recover darken image
        img = (img * 255.0).astype(np.uint8)
        # show
        img_concat = np.concatenate((img_ori, img), 1)
        cv2.imshow('img', img_concat)
        cv2.waitKey(0)
    
    def color_distortion_img_gen(img):
        img_ori = img.copy()
        img = img / 255.0
        # add darken
        img = color_distortion(img, prob = 1)
        # recover darken image
        img = (img * 255.0).astype(np.uint8)
        # show
        img_concat = np.concatenate((img_ori, img), 1)
        cv2.imshow('img', img_concat)
        cv2.waitKey(0)
    
    # read an image
    imgpath = 'F:\\submitted papers\\QuadBayer Deblur\\data\\val\\1_rgb_gt.png'
    img = cv2.imread(imgpath)
    h, w = img.shape[0], img.shape[1]
    #img = cv2.resize(img, (w // 3, h // 3))
    img = cv2.resize(img, (512, 512))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('original_img', img_gray)
    #cv2.waitKey(0)

    '''
    # RGB
    # K: 0.5, 0.25, 0.1; sigma: 10, 30
    img_concat = noisy_long_img_gen(img, K = 0.25, sigma = 5.0)
    cv2.imshow('img', img_concat)
    cv2.waitKey(0)

    img_concat = noisy_short_img_gen(img, K = 0.25, sigma = 5.0)
    cv2.imshow('img', img_concat)
    cv2.waitKey(0)
    '''
    
    # quadbayer
    short_pos = [[0,0], [1,1]]
    img_gray_one_fourth_quadbayer = img_gray.copy()
    img_gray_one_fourth_quadbayer = img_gray_one_fourth_quadbayer.astype(np.float64)
    for pos in short_pos:
        img_gray_one_fourth_quadbayer[pos[0]::2, pos[1]::2] /= 4
    img_concat_one_fourth_quadbayer1 = noisy_raw_gen(img_gray_one_fourth_quadbayer, K = 5, sigma = 5.0)
    img_concat_one_fourth_quadbayer2 = noisy_raw_gen(img_gray_one_fourth_quadbayer, K = 5, sigma = 1.0)
    for pos in short_pos:
        img_concat_one_fourth_quadbayer1[pos[0]::2, pos[1]::2] *= 4
        img_concat_one_fourth_quadbayer2[pos[0]::2, pos[1]::2] *= 4
    img_concat_one_fourth_quadbayer1 = img_concat_one_fourth_quadbayer1.astype(np.uint8)
    img_concat_one_fourth_quadbayer2 = img_concat_one_fourth_quadbayer2.astype(np.uint8)
    cv2.imwrite('img1.png', img_concat_one_fourth_quadbayer1)
    cv2.imwrite('img2.png', img_concat_one_fourth_quadbayer2)
    #cv2.imshow('img1', img_concat_one_fourth_quadbayer1)
    #cv2.waitKey(0)
    #cv2.imshow('img2', img_concat_one_fourth_quadbayer2)
    #cv2.waitKey(0)

    #img_concat_one_fourth_quadbayer = img_concat_one_fourth_quadbayer / 255.0
    #img_concat_one_fourth_quadbayer = img_concat_one_fourth_quadbayer ** (1 / 2.2)
    #img_concat_one_fourth_quadbayer = (img_concat_one_fourth_quadbayer * 255.0).astype(np.uint8)
    #cv2.imshow('img', img_concat_one_fourth_quadbayer)
    #cv2.waitKey(0)
    
    # short-exposure raw
    '''
    img_gray_one_fourth_quadbayer = img_gray.copy()
    img_gray_one_fourth_quadbayer = img_gray_one_fourth_quadbayer.astype(np.float64)
    img_gray_one_fourth_quadbayer /= 4
    img_concat_one_fourth_quadbayer = noisy_raw_gen(img_gray_one_fourth_quadbayer, K = 0.25, sigma = 5.0)
    img_concat_one_fourth_quadbayer *= 4
    img_concat_one_fourth_quadbayer = img_concat_one_fourth_quadbayer.astype(np.uint8)
    cv2.imshow('img', img_concat_one_fourth_quadbayer)
    cv2.waitKey(0)

    img_concat_one_fourth_quadbayer = img_concat_one_fourth_quadbayer / 255.0
    img_concat_one_fourth_quadbayer = img_concat_one_fourth_quadbayer ** (1 / 2.2)
    img_concat_one_fourth_quadbayer = (img_concat_one_fourth_quadbayer * 255.0).astype(np.uint8)
    cv2.imshow('img', img_concat_one_fourth_quadbayer)
    cv2.waitKey(0)
    '''
