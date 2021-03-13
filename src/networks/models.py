"""Model Script Used By train.py and predict.py"""
import torch
from torch import nn

from networks.eccv16 import eccv16

def get_model(model_name: str) -> nn.Module:
    """Get the nn.Module based on input"""
    if model_name == 'eccv16':
        return eccv16(pretrained=False, model_name=model_name)
    if model_name == 'eccv16_pretrained':
        return eccv16(pretrained=True, model_name=model_name)
    if model_name == 'eccv16_half':
        return eccv16(pretrained=False, hw_resize=128, model_name=model_name)
    raise ValueError('Unknown model')
