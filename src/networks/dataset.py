import os

from tqdm import tqdm
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

def kNN(ref, query, k: int = 10):
    """k-Nearest Neighbor algorithm in PyTorch
    Inputs:
        ref   : referenced training data. Shape : [n_samples, n_features]
        query : query points. Shape : [n_queries, n_features]
        k     : number of nearest neighbor to find. Default: 10
    Outputs:
        dists : the distances from each query point to its k-nearest neighbors.
                Shape : [n_queries, k]
        inds : the indices of the k-nearest neighbors for each query points.
                Shape : [n_queries, k]
    """
    dists = torch.norm(query[:,None,:] - ref, dim=-1)  # [n_queries, n_samples]
    # dists, inds = torch.sort(dists, dim=-1)
    # return dists[:, :k], inds[:, :k]
    dists, inds = torch.topk(dists, k, dim=-1, largest=False, sorted=False)
    return dists, inds

def get_dataset_prior_probs(root: str, annFile: str, batch_size: int,
                            ab_gamut_filepath: str,
                            device: torch.device):
    """Compute the prior probability of the quantized 313 ab values"""
    prior_probs_filepath = os.path.join(os.path.dirname(root),
                                        f'prior_probs_{os.path.basename(root)}.npy')

    if os.path.exists(prior_probs_filepath):
        print("Found existing prior probs, won't regenerate it")
        return np.load(prior_probs_filepath)

    print(f"Generating prior probability of dataset '{os.path.basename(root)}' ...")

    sigma = 5    # Sigma for Gaussian kernel

    dataset = load_dataset(root, annFile, batch_size)
    ab_grids = np.load(ab_gamut_filepath)
    ab_grids = torch.Tensor(ab_grids).to(device)

    prior_probs = torch.zeros((ab_grids.shape[0])).to(device)
    for batch_i, (_, img_ab) in enumerate(tqdm(dataset)):
        # Passing data to GPU
        img_ab = torch.moveaxis(img_ab, 1, -1).reshape(-1,2).to(device)

        # Find k-nearest neighbors
        dists, inds = kNN(ab_grids, img_ab)

        weights = torch.exp(-dists**2/(2*sigma**2))
        weights /= torch.sum(weights, dim=1)[:, None]

        for i in range(ab_grids.shape[0]):
            prior_probs[i] += torch.sum(weights[inds == i])
        # Free GPU memory
        del dists, inds, weights
    # Free memory
    del dataset, ab_grids, img_ab

    # Normalize and convert to numpy array
    prior_probs /= torch.sum(prior_probs)
    prior_probs = prior_probs.cpu().detach().numpy()

    # Save the prior probability
    np.save(prior_probs_filepath, prior_probs)

    return prior_probs

