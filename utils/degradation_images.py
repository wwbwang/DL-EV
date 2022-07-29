from sklearn.preprocessing import scale
import torch
import numpy as np
from skimage import filters
from skimage.util import random_noise, img_as_ubyte, img_as_float
from scipy.ndimage.interpolation import zoom as npzoom
from skimage.transform import rescale
import cv2
import matplotlib.pyplot as plt
import imageio

'''
 The additive-Gaussian model also outperformed the model trained with manually acquired training pairs (real world) across all metrics. 
 We further compared additive Gaussian with 'additive Gaussian (roughly 80x)', where we used approximately 80x more training data, which did not substantially increase the peak-signal-to-noise ratio (PSNR) or structural similarity (SSIM) measurements, but did further increase the resolution as measured by Fourier ring correlation (FRC) analysis.
'''

'''
# 
'''
def generate_image(img, opt):
    
    # img_ori_path = opt['generate images']['img_ori_path']
    # img_generate_folder = opt['generate images']['img_generate_folder']
    scale = opt['generate images']['scale']
    count = opt['generate images']['count']
    
    # img = cv2.imread(img_ori_path, cv2.IMREAD_UNCHANGED)
    img_ori = np.array(img)
    
    x_shape = img_ori.shape[0] // scale if img_ori.shape[0] % scale == 0 else img_ori.shape[0] // scale + 1
    y_shape = img_ori.shape[1] // scale if img_ori.shape[1] % scale == 0 else img_ori.shape[1] // scale + 1
    
    # img_list = []
    img_matrix = np.zeros((count, x_shape, y_shape))
    
    # First add noise and then downsample, repeat 5000 times
    for i in range(count):
        img_n = addnoise_localvar(img_ori)
        img_n = downsample(img_n, scale)
        # idx = '%04d' % i
        # imageio.imsave(img_generate_folder + '/' + idx + '.tif', img_n)
        img_matrix[i] = img_n
            
    return img_matrix

def addnoise_localvar(xn):
    if(xn.dtype == 'uint16'):
        datatype = np.uint16
    elif(xn.dtype == 'uint8'):
        datatype = np.uint8
    else:
        print('error datatype. The dataType is ' + xn.dtype) # TODO
    xorig_max = xn.max()
    xn = xn.astype(np.float32)
    xn /= float(np.iinfo(datatype).max)
    lvar = filters.gaussian(xn, sigma=5) + 1e-10
    xn = random_noise(xn, mode='localvar', local_vars=lvar*0.5)
    new_max = xn.max()
    if new_max > 0:
        xn /= new_max
    xn *= xorig_max
    return xn

def downsample(xn, scale=2):
    x_down = npzoom(xn, 1/scale, order=1)
    return x_down
