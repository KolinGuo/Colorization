#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Training Script"""
from datetime import datetime
import os
import io
import argparse
from typing import List
import argcomplete

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from networks.dataset import load_dataset
from networks.models import get_model
from networks.losses import get_loss_func
from networks.metrics import AUC

def get_parser() -> argparse.ArgumentParser:
    """Get the argparse parser for this script"""
    main_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Training\n\t')

    model_parser = main_parser.add_argument_group('Model configurations')
    model_parser.add_argument(
        'model',
        choices=['eccv16', 'eccv16_pretrained'],
        help="Network model used for training")

    dataset_parser = main_parser.add_argument_group('Dataset configurations')
    dataset_parser.add_argument(
        "--data-dir", type=str, default='/Colorization/data/train2017',
        help="Directory of COCO train2017 dataset (Default: data/train2017)")
    dataset_parser.add_argument(
        "--data-annFile", type=str, default='/Colorization/data/annotations/instances_train2017.json',
        help="Directory of COCO train2017 annotation file (Default: data/annotations/instances_train2017.json)")
    dataset_parser.add_argument(
        "--val-data-dir", type=str, default='/Colorization/data/val2017',
        help="Directory of COCO val2017 dataset (Default: data/val2017)")
    dataset_parser.add_argument(
        "--val-data-annFile", type=str, default='/Colorization/data/annotations/instances_val2017.json',
        help="Directory of COCO val2017 annotation file (Default: data/annotations/instances_val2017.json)")

    train_parser = main_parser.add_argument_group('Training configurations')
    train_parser.add_argument(
        '--loss-func',
        choices=['MSELoss'],
        default='MSELoss',
        help="Loss functions for training")
    train_parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size of patches (Default: 64)")
    train_parser.add_argument(
        "--num-epochs", type=int, default=20,
        help="Number of training epochs (Default: 20)")
    train_parser.add_argument(
        "--device", choices=['cuda:0', 'cpu'], default='cuda:0',
        help="Training device (Default: cuda:0)")
    #train_parser.add_argument(
    #    "--ckpt-weights-only", action='store_true',
    #    help="Checkpoints will only save the model weights (Default: False)")
    #train_parser.add_argument(
    #    "--ckpt-dir", type=str, default='/Colorization/checkpoints',
    #    help="Directory for saving/loading checkpoints")
    #train_parser.add_argument(
    #    "--ckpt-filepath", type=str, default=None,
    #    help="Checkpoint filepath to load and resume training from "
    #    "e.g. ./cp-001-50.51.ckpt.index")
    train_parser.add_argument(
        "--log-dir", type=str, default='/Colorization/tb_logs',
        help="Directory for saving tensorboard logs")
    train_parser.add_argument(
        "--file-suffix", type=str,
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Suffix for ckpt file and log file (Default: current timestamp)")

    #testing_parser = main_parser.add_argument_group('Testing configurations')
    #testing_parser.add_argument(
    #    "--steps-per-epoch", type=int, default=-1,
    #    help="Training steps per epoch (Testing only, don't modify)")
    #testing_parser.add_argument(
    #    "--val-steps", type=int, default=-1,
    #    help="Validation steps (Testing only, don't modify)")

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

    # Set tf.keras mixed precision to float16
    #set_keras_mixed_precision_policy('mixed_float16')

    # Load datasets
    train_dataset = load_dataset(args.data_dir, args.data_annFile,
                                 args.batch_size)
    val_dataset = load_dataset(args.val_data_dir, args.val_data_annFile,
                               args.batch_size)

    # Create network model
    model = get_model(args.model).to(pytorch_device)
    #model.summary(120)
    #print(keras.backend.floatx())

    # Get loss function
    loss_func = get_loss_func(args.loss_func, None)  # TODO: classweights
    # Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=3*1e-5,
                                 betas=(0.9,0.99),
                                 weight_decay=1e-3)
    # Evaluation metric
    auc_metric = AUC(step_size = 1.0, device=pytorch_device)



    #class_names = ['Background', 'Gray Matter', 'White Matter']
    #model.compile(optimizer=optimizers.Adam(),
    #              loss=get_loss_func(args.loss_func, class_weight,
    #                                 gamma=args.focal_loss_gamma),
    #              metrics=[metrics.SparseCategoricalAccuracy(),
    #                       SparseMeanIoU(num_classes=3, name='IoU/Mean'),
    #                       SparsePixelAccuracy(num_classes=3, name='PixelAcc'),
    #                       SparseMeanAccuracy(num_classes=3, name='MeanAcc'),
    #                       SparseFreqIoU(num_classes=3, name='IoU/Freq_weighted'),
    #                       SparseConfusionMatrix(num_classes=3, name='cm')] \
    #        + SparseIoU.get_iou_metrics(num_classes=3, class_names=class_names))

    # Create another checkpoint/log folder for model.name and timestamp
    #args.ckpt_dir = os.path.join(args.ckpt_dir,
    #                             model.name+'-'+args.file_suffix)
    args.log_dir = os.path.join(args.log_dir, 'fit',
                                model.name+'-'+args.file_suffix)

    ## Check if resume from training
    #initial_epoch = 0
    #if args.ckpt_filepath is not None:
    #    if args.ckpt_weights_only:
    #        if args.ckpt_filepath.endswith('.index'):   # Get rid of the suffix
    #            args.ckpt_filepath = args.ckpt_filepath.replace('.index', '')
    #        model.load_weights(args.ckpt_filepath).assert_existing_objects_matched()
    #        print('Model weights loaded')
    #    else:
    #        model = load_whole_model(args.ckpt_filepath)
    #        print('Whole model (weights + optimizer state) loaded')

    #    initial_epoch = int(args.ckpt_filepath.split('/')[-1]\
    #            .split('-')[1])
    #    # Save in same checkpoint_dir but different log_dir (add current time)
    #    args.ckpt_dir = os.path.abspath(
    #        os.path.dirname(args.ckpt_filepath))
    #    args.log_dir = args.ckpt_dir.replace(
    #        'checkpoints', 'tb_logs/fit') + f'-retrain_{args.file_suffix}'

    # Write configurations to log_dir
    log_configs(args.log_dir, args)

    # Create checkpoint directory
    #if not os.path.exists(args.ckpt_dir):
    #    os.makedirs(args.ckpt_dir)
    ## Create log directory
    #if not os.path.exists(args.log_dir):
    #    os.makedirs(args.log_dir)

    ## Create a callback that saves the model's weights every 1 epoch
    #if val_dataset:
    #    ckpt_path = os.path.join(
    #        args.ckpt_dir, 'cp-{epoch:03d}-{val_IoU/Mean:.4f}.ckpt')
    #else:
    #    ckpt_path = os.path.join(
    #        args.ckpt_dir, 'cp-{epoch:03d}-{IoU/Mean:.4f}.ckpt')
    #cp_callback = callbacks.ModelCheckpoint(
    #    filepath=ckpt_path,
    #    verbose=1,
    #    save_weights_only=args.ckpt_weights_only,
    #    save_freq='epoch')

    ## Create a TensorBoard callback
    #tb_callback = callbacks.TensorBoard(
    #    log_dir=args.log_dir,
    #    histogram_freq=1,
    #    write_graph=True,
    #    write_images=False,
    #    update_freq='batch',
    #    profile_batch='100, 120')

    ## Create a Lambda callback for plotting confusion matrix
    #cm_callback = get_cm_callback(args.log_dir, class_names)

    ## Create a TerminateOnNaN callback
    #nan_callback = callbacks.TerminateOnNaN()

    ## Create an EarlyStopping callback
    #if val_dataset:
    #    es_callback = callbacks.EarlyStopping(monitor='val_IoU/Mean',
    #                                          min_delta=0.01,
    #                                          patience=3,
    #                                          verbose=1,
    #                                          mode='max')

    #if val_dataset:
    #    model.fit(
    #        train_dataset,
    #        epochs=args.num_epochs,
    #        steps_per_epoch=len(train_dataset) \
    #                if args.steps_per_epoch == -1 else args.steps_per_epoch,
    #        initial_epoch=initial_epoch,
    #        validation_data=val_dataset,
    #        validation_steps=len(val_dataset) // args.val_subsplits \
    #                if args.val_steps == -1 else args.val_steps,
    #        callbacks=[cp_callback, tb_callback, nan_callback, cm_callback, es_callback])
    #else:
    #    model.fit(
    #        train_dataset,
    #        epochs=args.num_epochs,
    #        steps_per_epoch=len(train_dataset) \
    #                if args.steps_per_epoch == -1 else args.steps_per_epoch,
    #        initial_epoch=initial_epoch,
    #        callbacks=[cp_callback, tb_callback, nan_callback, cm_callback])
    # TODO: Switch to tf.data

    train_writer = SummaryWriter(args.log_dir + "/train")
    val_writer = SummaryWriter(args.log_dir + "/validation")
    batch_global_step = 0
    for epoch_i in range(args.num_epochs):
        # Train on training dataset
        epoch_loss = 0.0
        model = model.train()  # Set the module in training mode
        for batch_i, (img_l, img_ab) in enumerate(train_dataset):
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
        # Write epoch averaged loss
        train_epoch_loss = epoch_loss / (batch_i+1)
        train_writer.add_scalar('epoch_loss', train_epoch_loss,
                                global_step=epoch_i+1)

        # Evaluate on validation set
        epoch_val_loss = 0.0
        model = model.eval()  # Set the module in evaluation mode
        auc_metric.reset()    # Reset eval metric
        with torch.no_grad():
            for batch_i, (img_l, img_ab) in enumerate(val_dataset):
                # Passing data to GPU
                img_l  = img_l.to(pytorch_device)
                img_ab = img_ab.to(pytorch_device)

                # Forward pass through model
                img_ab_pred = model(img_l)

                # Compute loss
                loss = loss_func(img_ab_pred, img_ab)
                epoch_val_loss += loss
                # Update eval metric
                auc_metric.update((img_ab_pred, img_ab))

        # Write epoch averaged loss for validation set
        val_epoch_loss = epoch_val_loss / (batch_i+1)
        val_writer.add_scalar('epoch_loss', val_epoch_loss,
                              global_step=epoch_i+1)
        val_epoch_auc = auc_metric.compute()
        val_writer.add_scalar('epoch_auc', val_epoch_auc,
                              global_step=epoch_i+1)

        # Print loss/eval metric
        print(f'Epoch {epoch_i+1}/{args.num_epochs}: train_loss={train_epoch_loss}'
              f'\tval_loss={val_epoch_loss}\tval_auc={val_epoch_auc}')

        # TODO: Save Checkpoint

        # TODO: Print

    # Close the SummaryWriter
    train_writer.close()
    val_writer.close()

    print('Training finished!')

if __name__ == '__main__':
    parser = get_parser()
    argcomplete.autocomplete(parser)
    train_args = parser.parse_args()

    train(train_args)
