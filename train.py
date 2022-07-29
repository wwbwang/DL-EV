import cv2
import matplotlib.pyplot as plt
from utils.degradation_images import generate_image
from utils.test_model import re_model, get_operated_image
from utils.cal_stand_devi import cal_stand_devi
from utils import degradation_images, options
from arch import srresnet_arch
import numpy as np
import torch

def train_pipeline(root_path):
    
    opt, args = options.parse_options(root_path)
    opt['root_path'] = root_path
    upscale = opt['generate images']['scale']
    model_path = opt['evaluated model']['model_path']
    
    device = torch.device('cpu')
    model = srresnet_arch.MSRResNet(1, 1, 64, 16, upscale)
    model.load_state_dict(torch.load(model_path)['params_ema'], strict=True)
    model.eval()
    model = model.to(device)
    
    # get a single image use the given model
    img_LR = cv2.imread('img/LR_example.tif', cv2.IMREAD_UNCHANGED) 
    img = get_operated_image(img_LR, model, device)
    
    # get degraded images list
    img_matrix = generate_image(img, opt)
    
    # cal residual TODO
    
    # pass the degraded images through the given training model
    img_matrix = re_model(img_matrix, opt)
    
    # cal standard deviation
    standard_matrix = cal_stand_devi(img, img_matrix)
    print('mean of standard image is', np.mean(standard_matrix))
    
if __name__ == '__main__':
    import os
    root_path = os.getcwd()
    train_pipeline(root_path)
    
    # img = cv2.imread('cell.tif',cv2.IMREAD_UNCHANGED)
    # img = addnoise.addnoise_localvar(img)
    # plt.imshow(img)
    # print('1')