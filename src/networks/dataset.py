import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as dset
#import torchvision.transforms as transforms
from util import *

def input_transforms(img_rgb_orig, target, HW=(256,256), resample=3):
    # return resized L and ab channels as torch Tensors
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
    
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    img_l_rs = img_lab_rs[:,:,0]
    img_ab_rs = np.moveaxis(img_lab_rs[:,:,1:], -1, 0)  # (2, 256, 256)

    tens_rs_l = torch.Tensor(img_l_rs)[None,:,:]
    tens_rs_ab = torch.Tensor(img_ab_rs)[:,:]

    return (tens_rs_l, tens_rs_ab)

def load_dataset(root: str, annFile: str, batch_size: int):
    dataset = dset.CocoDetection(root=root,
                                 annFile=annFile,
                                 transforms=input_transforms)
    dataloader = data.DataLoader(dataset, batch_size=batch_size)
    return dataloader

