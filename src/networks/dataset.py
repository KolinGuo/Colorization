import os
import glob
from typing import Any, Callable, Optional, Tuple

from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data as data
import torchvision
torchvision.set_image_backend('accimage')
import torchvision.datasets as dset
import accimage
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
from util import *

def trainval_PIL_img_transform(img_rgb_orig, HW=(256,256), resample=3):
    # return resized L and ab channels as torch Tensors
    img_rgb_rs = resize_PIL_img(img_rgb_orig, HW=HW, resample=resample)
    
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    img_l_rs = img_lab_rs[:,:,0]
    img_ab_rs = np.moveaxis(img_lab_rs[:,:,1:], -1, 0)  # (2, 256, 256)

    tens_rs_l = torch.Tensor(img_l_rs)[None,:,:]
    tens_rs_ab = torch.Tensor(img_ab_rs)[:,:]

    return (tens_rs_l, tens_rs_ab)

def predict_PIL_img_transform(img_rgb_orig, HW=(256,256), resample=3):
    # return original size L and resized L as torch Tensors
    return preprocess_PIL_img(img_rgb_orig, HW=HW, resample=resample)

def trainval_acc_img_transform(img_path: str,
                               img_rgb_orig: np.ndarray,
                               img_rgb_rs: np.ndarray) -> Tuple[torch.Tensor]:
    # return resized L and ab channels as torch Tensors
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    img_l_rs = img_lab_rs[:,:,0]
    img_ab_rs = np.moveaxis(img_lab_rs[:,:,1:], -1, 0)  # (2, 256, 256)

    tens_rs_l = torch.Tensor(img_l_rs)[None,:,:]
    tens_rs_ab = torch.Tensor(img_ab_rs)[:,:]

    return (tens_rs_l, tens_rs_ab)

def predict_acc_img_transform(img_path: str,
                              img_rgb_orig: np.ndarray,
                              img_rgb_rs: np.ndarray) -> Tuple[torch.Tensor]:
    # return original size L and resized L as torch Tensors
    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    img_l_orig = img_lab_orig[:,:,0]
    img_l_rs = img_lab_rs[:,:,0]

    tens_orig_l = torch.Tensor(img_l_orig)[None,:,:]
    tens_rs_l = torch.Tensor(img_l_rs)[None,:,:]

    return (img_path, tens_orig_l, tens_rs_l)

class CocoColorization(dset.VisionDataset):
    """Coco Detection Dataset used for Colorization, use accimage instead of PIL"""
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None) -> None:
        self.root = os.path.abspath(root)
        super(CocoColorization, self).__init__(self.root, None, transform, None)
        self.img_paths = sorted(glob.glob(os.path.join(self.root, "*.jpg")))
        self.resize_trans = transforms.Resize(
                (256,256), interpolation=transforms.InterpolationMode.BICUBIC)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_path = self.img_paths[index]

        img = accimage.Image(img_path)   # accimage.Image
        img_arr = acc_image_to_np(img)
        img_rs_arr = acc_image_to_np(self.resize_trans.forward(img))
        return self.transform(img_path, img_arr, img_rs_arr)

        #img = Image.open(img_path).convert('RGB')     # PIL.Image
        #return self.transform(img)

    def __len__(self) -> int:
        return len(self.img_paths)

def load_trainval_dataset(root: str, annFile: str,
                          batch_size: int, shuffle: bool = True):
    dataset = CocoColorization(root, trainval_acc_img_transform)
    #dataset = CocoColorization(root, trainval_PIL_img_transform)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print(f'Train/Val Dataset "{os.path.basename(root)}" loaded: {len(dataset)} samples,',
          f'{len(dataloader)} batches')
    return dataloader

def load_predict_dataset(root: str, annFile: str,
                         batch_size: int = 1, shuffle: bool = False):
    dataset = CocoColorization(root, predict_acc_img_transform)
    #dataset = CocoColorization(root, predict_PIL_img_transform)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print(f'Predict Dataset "{os.path.basename(root)}" loaded: {len(dataset)} samples,',
          f'{len(dataloader)} batches')
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

class PriorFactor():
    """Prior Factor w class"""
    def __init__(self, prior_probs: np.ndarray, alpha=1.0, lam=0.0, verbose=False):
        """
        Inputs:
            prior_probs: prior probability array.
            alpha : prior factor inverse power.
            lam : percentage to mix in uniform prior with empirical prior
        """
        self.prior_probs = prior_probs
        self.alpha = alpha
        self.lam = lam
        self.verbose = verbose

        # define uniform probability
        self.uni_probs = np.zeros_like(self.prior_probs)
        self.uni_probs[self.prior_probs != 0] = 1.
        self.uni_probs = self.uni_probs/np.sum(self.uni_probs)

        # convex combination of empirical prior and uniform distribution
        self.prior_mix = (1-self.lam)*self.prior_probs + self.lam*self.uni_probs

        # set prior factor
        self.prior_factor = self.prior_mix**-self.alpha
        self.prior_factor = self.prior_factor/np.sum(self.prior_probs*self.prior_factor) # re-normalize

        if(self.verbose):
            self.print_correction_stats()

    def print_correction_stats(self):
        print('Prior factor correction:')
        print('  (alpha, lam) = (%.2f, %.2f)'%(self.alpha,self.lam))
        print('  (min,max,mean,med,exp) = (%.2f, %.2f, %.2f, %.2f, %.2f)' \
                % (np.min(self.prior_factor),
                   np.max(self.prior_factor),
                   np.mean(self.prior_factor),
                   np.median(self.prior_factor),
                   np.sum(self.prior_factor*self.prior_probs)))

    def forward(self, data_ab_quant, axis=1):
        """Retrieve prior_factor of the input quantized ab data
        Inputs:
            data_ab_quant : quantized ab data. Shape: [batch_size, Q, H, W]
            axis : the axis where the ab channel is at. Default: 1
        Output:
            corr_factor : the prior correction factors. Shape: [batch_size, H, W]
        """
        if data_ab_quant.ndim == 4:
            data_ab_maxind = np.argmax(data_ab_quant,axis=axis)  # [batch_size, H, W]
        else:
            data_ab_maxind = data_ab_quant
        corr_factor = self.prior_factor[data_ab_maxind]  # [batch_size, H, W]
        #return np.expand_dims(corr_factor, axis)  # [batch_size, 1, H, W]
        return corr_factor  # [batch_size, H, W]

class ColorQuantization():
    """Color Quantization Class"""
    def __init__(self, k: int = 10, sigma: float = 5, ab_gamut_filepath: str = ''):
        """
        Inputs:
            k : number of nearest neighbor to find.
            sigma : standard deviation of Gaussian kernel.
            ab_gamut_filepath : filepath of sRGB in-gamut file of ab space.
        """
        self.k = k
        self.sigma = sigma
        self.ab_grids = np.load(ab_gamut_filepath)  # [313, 2]
        self.Q = self.ab_grids.shape[0]
        self.knn_model = NearestNeighbors(n_neighbors=k,
                                          algorithm='ball_tree',
                                          n_jobs=-1).fit(self.ab_grids)

    def encode(self, ab_true: np.ndarray, axis: int = 1) -> np.ndarray:
        """Quantize the ab_true groundtruth values with soft-encoding
        Input:
            ab_true : groundtruth ab channel values. Shape: [batch_size, 2, H, W]
            axis : axis where the ab channel is at.
        """

        raise NotImplementedError('encode not implemented yet')

    def encode_1hot(self, ab_data: np.ndarray, axis: int = 1) -> np.ndarray:
        """Quantize the ab_true groundtruth values with one-hot encoding
        Inputs:
            ab_data : ab channel values. Shape: [batch_size, 2, H, W]
            axis : axis where the ab channel is at.
        Output:
            ab_data_minind : encoded closest points indices of ab_grid.
        """
        ab_data = np.moveaxis(ab_data, axis, -1)  # [batch_size, H, W, 2]
        prev_shape = np.array(ab_data.shape[:-1]).tolist()  # [batch_size, H, W]
        ab_data = ab_data.reshape(-1, ab_data.shape[-1])  # [batch_size * H*W, 2]

        # Compute norm of difference
        dists = np.linalg.norm(ab_data[:,None,:] - self.ab_grids, axis=-1)  # [batch_size * H*W, Q]
        ab_data_minind = np.argmin(dists, axis=-1)  # [batch_size * H*W,]

        # Reshape back to prev_shape
        ab_data_minind = ab_data_minind.reshape(prev_shape)  # [batch_size, H, W]
        return ab_data_minind
