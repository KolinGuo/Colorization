from typing import Tuple

from PIL import Image
#import accimage
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F
from IPython import embed

def load_img(img_path):
    out_np = np.asarray(Image.open(img_path))
    if(out_np.ndim==2):
        out_np = np.tile(out_np[:,:,None],3)
    return out_np

def acc_image_to_np(image: "accimage.Image") -> np.ndarray:
    """
    Returns:
        np.ndarray: Image converted to array with shape (H, W, channels)
    """
    image_np = np.empty([image.channels, image.height, image.width], dtype=np.uint8)
    image.copyto(image_np)
    return np.moveaxis(image_np, 0, -1)  # [H, W, 3]

def resize_PIL_img(img: Image, HW=(256,256), resample=3) -> np.ndarray:
    #return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))
    return np.asarray(img.resize((HW[1],HW[0]), resample=resample))

def preprocess_PIL_img(img_rgb_orig: Image, HW=(256,256), resample=3) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    # return original size L and resized L as torch Tensors
    img_rgb_orig_arr = np.asarray(img_rgb_orig)
    img_rgb_rs_arr = resize_PIL_img(img_rgb_orig, HW=HW, resample=resample)
    
    img_lab_orig = color.rgb2lab(img_rgb_orig_arr)
    img_lab_rs = color.rgb2lab(img_rgb_rs_arr)

    img_l_orig = img_lab_orig[:,:,0]
    img_l_rs = img_lab_rs[:,:,0]

    tens_orig_l = torch.Tensor(img_l_orig)[None,:,:]
    tens_rs_l = torch.Tensor(img_l_rs)[None,:,:]

    return (tens_orig_l, tens_rs_l)

#def resize_acc_img(img: "accimage.Image", HW=(256,256), resample=3) -> np.ndarray:
#    """Resize an accimage.Image"""
#    img = img.resize((HW[1],HW[0]))
#    return acc_image_to_np(img)
#
#def preprocess_acc_img(img_rgb_orig: "accimage.Image", HW=(256,256), resample=3) \
#        -> Tuple[torch.Tensor, torch.Tensor]:
#    # return original size L and resized L as torch Tensors
#    img_rgb_orig_arr = acc_image_to_np(img_rgb_orig)
#    img_rgb_rs_arr = resize_acc_img(img_rgb_orig, HW=HW, resample=resample)
#
#    img_lab_orig = color.rgb2lab(img_rgb_orig_arr)
#    img_lab_rs = color.rgb2lab(img_rgb_rs_arr)
#
#    img_l_orig = img_lab_orig[:,:,0]
#    img_l_rs = img_lab_rs[:,:,0]
#
#    tens_orig_l = torch.Tensor(img_l_orig)[None,:,:]
#    tens_rs_l = torch.Tensor(img_l_rs)[None,:,:]
#
#    return (tens_orig_l, tens_rs_l)

def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
    # tens_orig_l     1 x 1 x H_orig x W_orig
    # out_ab         1 x 2 x H x W

    HW_orig = tens_orig_l.shape[2:]
    HW = out_ab.shape[2:]

    # call resize function if needed
    if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
    else:
        out_ab_orig = out_ab

    out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
    out_rgb_orig = color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))
    return (255 * np.clip(out_rgb_orig, 0, 1)).astype('uint8')
