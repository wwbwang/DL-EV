from ast import Return
from itertools import count
from arch import srresnet_arch
import torch
import numpy as np
import imageio

def re_model(img_matrix, opt):
    arch_path = opt['evaluated model']['arch_path']
    model_path = opt['evaluated model']['model_path']
    count = opt['generate images']['count']
    upscale = opt['generate images']['scale']
    
    return train_model(img_matrix, arch_path, model_path, count, upscale)
    

def train_model(img_matrix, arch_path, model_path, count, upscale):
    # _dataset_modules = [importlib.import_module(f'basicsr.data.{file_name}') for file_name in dataset_filenames]
    device = torch.device('cpu')
    model = srresnet_arch.MSRResNet(1, 1, 64, 16, upscale)
    model.load_state_dict(torch.load(model_path)['params_ema'], strict=True)

    model.eval()
    model = model.to(device)
    
    output_matrix = np.zeros((count, img_matrix.shape[1]*upscale, img_matrix.shape[2]*upscale))
    
    for i in range(count):
        img = img_matrix[i]
        output_matrix[i] = get_operated_image(img, model, device)
        
    return output_matrix

def get_operated_image(img, model, device):
    img = np.expand_dims(img, axis = 0)
    img = img * 1.0 / 65535 
    img = torch.from_numpy(img).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = (output * 65535.0).round()
    output = output.astype(np.uint16)   #转为unit16
    return output