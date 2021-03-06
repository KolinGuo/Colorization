#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Training Script"""
from datetime import datetime
import os
import io
import argparse
from typing import List
import argcomplete

from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from networks.dataset import load_trainval_dataset, load_predict_dataset, \
                             get_dataset_prior_probs, ColorQuantization
from networks.models import get_model
from networks.losses import get_loss_func
from networks.metrics import get_metric
from util import postprocess_tens

def get_parser() -> argparse.ArgumentParser:
    """Get the argparse parser for this script"""
    main_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Training\n\t')

    model_parser = main_parser.add_argument_group('Model configurations')
    model_parser.add_argument(
        'model',
        choices=['eccv16', 'eccv16_pretrained', 'eccv16_half'],
        help="Network model used for training")

    dataset_parser = main_parser.add_argument_group('Dataset configurations')
    dataset_parser.add_argument(
        "--data-dir", type=str, default='/Colorization/data/train2017',
        help="Directory of COCO train2017 dataset (Default: data/train2017)")
    dataset_parser.add_argument(
        "--data-annFile", type=str, default='/Colorization/data/annotations/instances_train2017.json',
        help="COCO train2017 annotation file (Default: data/annotations/instances_train2017.json)")
    dataset_parser.add_argument(
        "--ab-gamut-file", type=str, default='/Colorization/data/pts_in_hull.npy',
        help="sRGB in-gamut file of ab space (Default: data/pts_in_hull.npy)")
    dataset_parser.add_argument(
        "--val-data-dir", type=str, default='/Colorization/data/val2017',
        help="Directory of COCO val2017 dataset (Default: data/val2017)")
    dataset_parser.add_argument(
        "--val-data-annFile", type=str, default='/Colorization/data/annotations/instances_val2017.json',
        help="COCO val2017 annotation file (Default: data/annotations/instances_val2017.json)")

    train_parser = main_parser.add_argument_group('Training configurations')
    train_parser.add_argument(
        '--loss-func',
        choices=['MSELoss', 'MSELoss_Vibrant'],
        default='MSELoss_Vibrant',
        help="Loss functions for training")
    train_parser.add_argument(
        '--eval-metric',
        choices=['AUC', 'RebalancedAUC'],
        default='RebalancedAUC',
        help="Evaluation metric on the validation dataset")
    train_parser.add_argument(
        '-vi', '--val-img-idx', type=int, action='append',
        default=[1, 2, 3, 14, 30, 80],  # To append: -vi 10 -vi 20
        help="Image indices from validation dataset to visualize for each epoch")
    train_parser.add_argument(
        "--color-vivid-gamma", type=float, default=2.0,
        help="Color vividness loss weight gamma (Default: 2.0)")
    train_parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size of patches (Default: 64)")
    train_parser.add_argument(
        "--val-batch-size", type=int, default=128,
        help="Batch size of patches in validation (Default: 128)")
    train_parser.add_argument(
        "--num-epochs", type=int, default=20,
        help="Number of training epochs (Default: 20)")
    train_parser.add_argument(
        "--device", choices=['cuda:0', 'cpu'], default='cuda:0',
        help="Training device (Default: cuda:0)")
    #train_parser.add_argument(
    #    "--ckpt-weights-only", action='store_true',
    #    help="Checkpoints will only save the model weights (Default: False)")
    train_parser.add_argument(
        "--ckpt-dir", type=str, default='/Colorization/checkpoints',
        help="Directory for saving/loading checkpoints")
    train_parser.add_argument(
        "--ckpt-filepath", type=str, default=None,
        help="Checkpoint filepath to load and resume training from "
             "e.g. ./cp-001-50.51.ckpt.pth")
    train_parser.add_argument(
        "--log-dir", type=str, default='/Colorization/tb_logs',
        help="Directory for saving tensorboard logs")
    train_parser.add_argument(
        "--file-suffix", type=str,
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Suffix for ckpt file and log file (Default: current timestamp)")

    testing_parser = main_parser.add_argument_group('Testing configurations')
    testing_parser.add_argument(
        "--batch-per-epoch", type=int, default=-1,
        help="Training batches per epoch (Testing only, don't modify)")
    testing_parser.add_argument(
        "--val-batch", type=int, default=-1,
        help="Validation batches (Testing only, don't modify)")

    return main_parser

def log_configs(log_dir: str, args) -> None:
    """Log configuration of this training session"""
    #writer = tf.summary.create_file_writer(log_dir + "/config")
    writer = SummaryWriter(log_dir + "/config")

    for key, value in vars(args).items():
        writer.add_text(str(key), str(value), global_step=0)

    writer.flush()
    writer.close()  # FIXME: Cautious, unused in TF

def train(args) -> None:
    """Start training based on args input"""
    # Check if GPU is available
    print("\nNum GPUs Available: %d\n"\
          % (torch.cuda.device_count()))
    # Set pytorch_device
    pytorch_device = torch.device(args.device)

    # Compute prior probability
    prior_probs = get_dataset_prior_probs(args.data_dir, args.data_annFile,
                                          args.batch_size // 4,
                                          args.ab_gamut_file, pytorch_device)
    # Empty GPU cache if possible
    torch.cuda.empty_cache()

    # Resizing factor
    if args.model == 'eccv16' or args.model == 'eccv16_pretrained':
        args.hw_resize = 256
    if args.model == 'eccv16_half':
        args.hw_resize = 128

    # Load datasets
    train_dataset = load_trainval_dataset(
            args.data_dir, args.data_annFile,
            args.batch_size, hw_resize=args.hw_resize, shuffle=True)
    val_dataset = load_trainval_dataset(
            args.val_data_dir, args.val_data_annFile,
            args.val_batch_size, hw_resize=args.hw_resize, shuffle=False)
    args.val_img_idx = sorted(args.val_img_idx)
    val_pred_dataset = load_predict_dataset(
            args.val_data_dir, None, subset_idx=args.val_img_idx,
            batch_size=1, hw_resize=args.hw_resize, shuffle=False)

    # Create network model
    model = get_model(args.model).to(pytorch_device)
    #model.summary(120)
    #print(keras.backend.floatx())

    # Get loss function
    loss_func = get_loss_func(args.loss_func, None,
                              color_vivid_gamma=args.color_vivid_gamma)  # TODO: classweights
    # Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=3*1e-5,
                                 betas=(0.9,0.99),
                                 weight_decay=1e-3)
    # Evaluation metric
    color_quant = ColorQuantization(k=1, sigma=5,
                                    ab_gamut_filepath=args.ab_gamut_file)
    #auc_metric = AUC(step_size = 1.0, device=pytorch_device)  # CUDA: Out-of-memory
    metric = get_metric(args.eval_metric, prior_probs, color_quant, step_size=1.0)

    # Create another checkpoint/log folder for model.name and timestamp
    args.ckpt_dir = os.path.join(args.ckpt_dir,
                                 model.name+'-'+args.file_suffix)
    args.log_dir = os.path.join(args.log_dir, 'fit',
                                model.name+'-'+args.file_suffix)

    # Check if resume from training
    start_epoch = 0
    if args.ckpt_filepath is not None:
        ckpt = torch.load(args.ckpt_filepath)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        print(f'Model weights loaded, starting epoch = {start_epoch}/{args.num_epochs}')

        # Save in same checkpoint_dir but different log_dir (add current time)
        args.ckpt_dir = os.path.abspath(
            os.path.dirname(args.ckpt_filepath))
        args.log_dir = args.ckpt_dir.replace(
            'checkpoints', 'tb_logs/fit') + f'-retrain_{args.file_suffix}'

    # Write configurations to log_dir
    log_configs(args.log_dir, args)

    # Create checkpoint directory
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    ## Create log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Create Tensorboard SummaryWriter
    train_writer = SummaryWriter(args.log_dir + "/train")
    val_writer = SummaryWriter(args.log_dir + "/validation")

    # Begin training
    batch_global_step = 0
    for epoch_i in range(start_epoch, args.num_epochs):
        # Train on training dataset
        epoch_loss = 0.0
        model = model.train()  # Set the module in training mode
        for batch_i, (img_l, img_ab) in enumerate(tqdm(train_dataset)):
            # Passing data to GPU
            img_l  = img_l.to(pytorch_device)
            img_ab = img_ab.to(pytorch_device)

            # Forward pass through model
            img_ab_pred = model(img_l)

            # Compute loss
            loss = loss_func(img_ab_pred, img_ab)
            batch_global_step += 1
            epoch_loss += loss
            # Write batch loss
            train_writer.add_scalar('batch_loss', loss,
                                    global_step=batch_global_step)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Preliminary stop for debugging
            if args.batch_per_epoch > 0 and batch_i+1 == args.batch_per_epoch:
                break

        # Write epoch averaged loss
        train_epoch_loss = epoch_loss / (batch_i+1)
        train_writer.add_scalar('epoch_loss', train_epoch_loss,
                                global_step=epoch_i+1)

        # Evaluate on validation set
        epoch_val_loss = 0.0
        model = model.eval()  # Set the module in evaluation mode
        metric.reset()    # Reset eval metric
        with torch.no_grad():
            for batch_i, (img_l, img_ab) in enumerate(tqdm(val_dataset)):
                # Passing data to GPU
                img_l  = img_l.to(pytorch_device)
                img_ab = img_ab.to(pytorch_device)

                # Forward pass through model
                img_ab_pred = model(img_l)

                # Compute loss
                loss = loss_func(img_ab_pred, img_ab)
                epoch_val_loss += loss
                # Update eval metric
                metric.update((img_ab_pred, img_ab))

                # Preliminary stop for debugging
                if args.val_batch > 0 and batch_i+1 == args.val_batch:
                    break

        # Write epoch averaged loss for validation set
        val_epoch_loss = epoch_val_loss / (batch_i+1)
        val_writer.add_scalar('epoch_loss', val_epoch_loss,
                              global_step=epoch_i+1)
        val_epoch_auc = metric.compute()
        val_writer.add_scalar(f'epoch_{args.eval_metric}', val_epoch_auc,
                              global_step=epoch_i+1)

        # Print loss/eval metric
        print(f'Epoch {epoch_i+1}/{args.num_epochs}: train_loss={train_epoch_loss}'
              f'\tval_loss={val_epoch_loss}\tval_auc={val_epoch_auc:.4f}')

        # Generate colorized validation images and visualize in Tensorboard
        model = model.eval()  # Set the module in evaluation mode
        with torch.no_grad():
            for batch_i, (img_path, tens_orig_l, tens_rs_l) in enumerate(tqdm(val_pred_dataset)):
                img_name = os.path.basename(img_path[0]).split('.')[0]  # '000000002532'
                # Passing data to GPU
                tens_rs_l = tens_rs_l.to(pytorch_device)
                # Forward pass through model
                tens_rs_ab_pred = model(tens_rs_l).cpu()
                # Post processing and Write to Tensorboard
                if epoch_i == 0:
                    img_gray = postprocess_tens(tens_orig_l, torch.cat((0*tens_orig_l,0*tens_orig_l),dim=1))
                    img_orig = np.asarray(Image.open(img_path[0]).convert('RGB'))
                    val_writer.add_images(img_name+'_gray_true.png',
                                          np.stack((img_gray, img_orig), axis=0).astype('uint8'),
                                          global_step=0, dataformats='NHWC')
                img_pred = postprocess_tens(tens_orig_l, tens_rs_ab_pred)
                val_writer.add_image(img_name+'_pred.png', img_pred,
                                     global_step=epoch_i+1, dataformats='HWC')

        # Save Checkpoint
        ckpt_path = os.path.join(args.ckpt_dir, 
                                 f'cp-{epoch_i+1:03d}-{val_epoch_auc:.4f}.ckpt.pth')
        torch.save({
            'epoch': epoch_i+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_epoch_loss': train_epoch_loss,
            'val_epoch_loss': val_epoch_loss,
            'val_epoch_auc': val_epoch_auc,
            }, ckpt_path)
        print(f'Checkpoint saved to {ckpt_path}')

    # Close the SummaryWriter
    train_writer.close()
    val_writer.close()

    print('Training finished!')

if __name__ == '__main__':
    parser = get_parser()
    argcomplete.autocomplete(parser)
    train_args = parser.parse_args()

    # Ignore skimage.color.colorconv.lab2xyz() out-of-range warnings
    import warnings
    warnings.filterwarnings("ignore", message='Color data out of range.+',
                            category=UserWarning,
                            module='skimage.color.colorconv',
                            lineno=1128, append=False)

    train(train_args)
