#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Predicting Script"""
import os
import gc
import argparse
from typing import List
import argcomplete

from tqdm import tqdm
import numpy as np
from PIL import Image
import torch

from networks.dataset import load_predict_dataset, get_dataset_prior_probs, \
                             ColorQuantization
from networks.models import get_model
from util import postprocess_tens

def get_parser() -> argparse.ArgumentParser:
    """Get the argparse parser for this script"""
    main_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Predicting\n\t')

    ckpt_parser = main_parser.add_argument_group('Model checkpoint configurations')
    ckpt_parser.add_argument(
        'model',
        choices=['eccv16', 'eccv16_pretrained'],
        help="Network model used for predicting")
    ckpt_parser.add_argument(
        "--ckpt-filepath", type=str, default=None,
        help="Checkpoint filepath to load and predict from "
             "e.g. ./cp-001-50.51.ckpt.pth")

    dataset_parser = main_parser.add_argument_group('Dataset configurations')
    dataset_parser.add_argument(
        "--data-dir", type=str, default='/Colorization/data/val2017',
        help="Directory of folder of jpg images (Default: data/val2017)")
    #dataset_parser.add_argument(
    #    "--data-annFile", type=str, default='/Colorization/data/annotations/instances_val2017.json',
    #    help="COCO val2017 annotation file (Default: data/annotations/instances_val2017.json)")

    predict_parser = main_parser.add_argument_group('Predicting configurations')
    predict_parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size of patches")
    predict_parser.add_argument(
        "--save-dir", type=str, default="/Colorization/data/outputs",
        help="Output directory (Default: /Colorization/data/outputs/model_name_cp-001-0.4584)")
    predict_parser.add_argument(
        "--device", choices=['cuda:0', 'cpu'], default='cuda:0',
        help="Predicting device (Default: cuda:0)")

    #testing_parser = main_parser.add_argument_group('Testing configurations')
    #testing_parser.add_argument(
    #    "--test-img-idx", type=int, default=-1,
    #    help="Test predicting image index (Testing only, don't modify)")
    #testing_parser.add_argument(
    #    "--predict-one-round", action='store_true',
    #    help="Only predict one round (Default: False; Testing only, don't modify)")

    return main_parser

def predict(args):
    """Start predicting based on args input"""
    # Check if GPU is available
    print("\nNum GPUs Available: %d\n"\
          % (torch.cuda.device_count()))
    # Set pytorch_device
    pytorch_device = torch.device(args.device)

    # Load dataset
    dataset = load_predict_dataset(
            args.data_dir, None,
            args.batch_size, shuffle=False)

    # Create network model
    model = get_model(args.model).to(pytorch_device)
    if args.ckpt_filepath is not None:
        ckpt = torch.load(args.ckpt_filepath)
        model.load_state_dict(ckpt['model_state_dict'])
        model.name += '_' + os.path.basename(args.ckpt_filepath).split('.ckpt.pth')[0]
        print(f'Model "{model.name}" weights loaded')

    # Create output directory
    args.save_dir = os.path.join(os.path.abspath(args.save_dir), model.name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print(f'\tSaving predicted mask to "{args.save_dir}"')

    # Begin predicting
    model = model.eval()  # Set the module in evaluation mode
    with torch.no_grad():
        for batch_i, (img_path, tens_orig_l, tens_rs_l) in enumerate(tqdm(dataset)):
            img_name = os.path.basename(img_path[0]).split('.')[0]
            # Passing data to GPU
            tens_rs_l = tens_rs_l.to(pytorch_device)

            # Forward pass through model
            tens_rs_ab_pred = model(tens_rs_l).cpu()

            # Post processing and save as images
            img_gray = postprocess_tens(tens_orig_l, torch.cat((0*tens_orig_l,0*tens_orig_l),dim=1))
            img_pred = postprocess_tens(tens_orig_l, tens_rs_ab_pred)
            img_gray = Image.fromarray(img_gray)
            img_pred = Image.fromarray(img_pred)
            img_gray.save(os.path.join(args.save_dir, img_name+'_gray.png'))
            img_pred.save(os.path.join(args.save_dir, img_name+'_pred.png'))

    print('Predicting finished!')


if __name__ == '__main__':
    parser = get_parser()
    argcomplete.autocomplete(parser)
    predict_args = parser.parse_args()

    # Ignore skimage.color.colorconv.lab2xyz() out-of-range warnings
    import warnings
    warnings.filterwarnings("ignore", message='Color data out of range.+',
                            category=UserWarning,
                            module='skimage.color.colorconv',
                            lineno=1128, append=False)

    predict(predict_args)
