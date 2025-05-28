"""
FiCo-ITR integration script for Query2Label.
This script should be placed in the root directory of the Query2Label repository.

Generates semantic category labels for Flickr30K dataset using Q2L model.
"""

import argparse
import os, sys
import random
import datetime
import time
from typing import List
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import _init_paths
from dataset.get_dataset import get_datasets
from dataset.flickrdataset import Flickr30kDataset
from torchvision import transforms


from utils.logger import setup_logger
import models
import models.aslloss
from models.query2label import build_q2l
from utils.metric import voc_mAP
from utils.misc import clean_state_dict
from utils.slconfig import get_raw_dict
from tqdm import tqdm


def parser_args():
    available_models = ['Q2L-R101-448', 'Q2L-R101-576', 'Q2L-TResL-448', 'Q2L-TResL_22k-448', 'Q2L-SwinL-384', 'Q2L-CvT_w24-384']

    parser = argparse.ArgumentParser(description='Query2Label Flickr30K Label Generation for FiCo-ITR')
    parser.add_argument('--dataset_dir', help='Directory containing Flickr30K images', required=True)
    parser.add_argument('--output_labels', help='Output path for labels .npy file', 
                        default='flickr30k_labels.npy')
    parser.add_argument('--output_names', help='Output path for image names .txt file', 
                        default='flickr30k_names.txt')
    
    parser.add_argument('--img_size', default=448, type=int,
                        help='image size. default(448)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='Q2L-R101-448',
                        choices=available_models,
                        help='model architecture: ' +
                            ' | '.join(available_models) +
                            ' (default: Q2L-R101-448)')
    parser.add_argument('--config', type=str, help='config file')

    parser.add_argument('--num_class', default=80, type=int,
                        help="Number of classes (80 for COCO categories)")
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N',
                        help='mini-batch size (default: 16), this is the total '
                            'batch size of all GPUs')
    parser.add_argument('--resume', type=str, metavar='PATH', required=True,
                        help='path to Q2L checkpoint')

    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model. default is False. ')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='threshold for positive label (default: 0.5)')

    # Transformer parameters (using Q2L defaults)
    parser.add_argument('--enc_layers', default=1, type=int, 
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=256, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=128, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--keep_other_self_attn_dec', action='store_true', 
                        help='keep the other self attention modules in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_first_self_attn_dec', action='store_true',
                        help='keep the first self attention module in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_input_proj', action='store_true', 
                        help="keep the input projection layer. Needed when the channel of image features is different from hidden_dim of Transformer layers.")

    # Basic setup
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='device to use for inference')
    
    args = parser.parse_args()

    # update parameters with pre-defined config file
    if args.config:
        with open(args.config, 'r') as f:
            cfg_dict = json.load(f)
        for k,v in cfg_dict.items():
            setattr(args, k, v)

    return args


def main():
    args = parser_args()
    
    # Set random seeds for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Set device
    device = torch.device(args.device)
    if args.device == 'cuda':
        cudnn.benchmark = True
    
    # Build model
    print("Building Q2L model...")
    model = build_q2l(args)
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device)
    state_dict = clean_state_dict(checkpoint['state_dict'])
    model.load_state_dict(state_dict, strict=True)
    print("Checkpoint loaded successfully")
    
    # Set to evaluation mode
    model.eval()
    
    # Data transform
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])

    # Create dataset and dataloader
    print(f"Loading Flickr30K dataset from {args.dataset_dir}")
    flickr30k_dataset = Flickr30kDataset(root_dir=args.dataset_dir, transform=transform)
    flickr30k_loader = torch.utils.data.DataLoader(
        flickr30k_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.workers, 
        pin_memory=True
    )
    
    print(f"Found {len(flickr30k_dataset)} images")
    
    # Generate labels
    saved_labels, saved_names = generate_labels(flickr30k_loader, model, args, device)
    
    # Save results
    print(f"Saving labels to {args.output_labels}")
    np.save(args.output_labels, saved_labels)
    print(f"Labels shape: {saved_labels.shape}")
    
    print(f"Saving image names to {args.output_names}")
    with open(args.output_names, 'w') as f:
        for name in saved_names:
            f.write(f"{name}\n")
    print(f"Saved {len(saved_names)} image names")
    
    # Print statistics
    labels_per_image = saved_labels.sum(axis=1)
    print(f"\nLabel statistics:")
    print(f"  Min labels per image: {labels_per_image.min()}")
    print(f"  Max labels per image: {labels_per_image.max()}")
    print(f"  Mean labels per image: {labels_per_image.mean():.2f}")
    print(f"  Images with no labels: {(labels_per_image == 0).sum()}")
    
    print("\nDone!")


@torch.no_grad()
def generate_labels(val_loader, model, args, device):
    """Generate multi-label predictions for all images."""
    
    saved_labels = []
    saved_names = []
    
    with torch.no_grad():
        for i, (images, names) in enumerate(tqdm(val_loader, desc='Generating labels')):
            images = images.to(device, non_blocking=True)

            # Forward pass
            output = model(images)
            output_sm = nn.functional.sigmoid(output)

            # Apply threshold
            binary_labels = (output_sm >= args.threshold).int()
            saved_labels.append(binary_labels.cpu().numpy())
            saved_names.extend(names)

    return np.concatenate(saved_labels, axis=0), saved_names


if __name__ == '__main__':
    main()