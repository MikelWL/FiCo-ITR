#!/usr/bin/env python3
"""
Extract 4096D VGG-19 features for coarse-grained models.

This tool extracts VGG-19 features from images, commonly used in coarse-grained
retrieval models. Features are extracted from the fc2 layer (4096-dimensional).
"""

import argparse
import json
import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ImageDataset(Dataset):
    """Custom dataset for loading images."""
    
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        return image, idx


def load_image_list(list_file, image_dir=None):
    """Load list of image paths from file.
    
    Supports multiple formats:
    - txt: One image path per line
    - json: List of image paths or dict with image paths
    - jsonl: One JSON object per line with 'image' or 'path' field
    """
    image_paths = []
    
    ext = os.path.splitext(list_file)[1].lower()
    
    if ext == '.txt':
        with open(list_file, 'r') as f:
            for line in f:
                path = line.strip()
                if path:
                    if image_dir and not os.path.isabs(path):
                        path = os.path.join(image_dir, path)
                    image_paths.append(path)
    
    elif ext == '.json':
        with open(list_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                # List of paths
                for path in data:
                    if isinstance(path, str):
                        if image_dir and not os.path.isabs(path):
                            path = os.path.join(image_dir, path)
                        image_paths.append(path)
                    elif isinstance(path, dict):
                        # Try common field names
                        for field in ['image', 'path', 'file_name', 'filename']:
                            if field in path:
                                p = path[field]
                                if image_dir and not os.path.isabs(p):
                                    p = os.path.join(image_dir, p)
                                image_paths.append(p)
                                break
            elif isinstance(data, dict):
                # Dict mapping IDs to paths
                for img_id, path in data.items():
                    if isinstance(path, str):
                        if image_dir and not os.path.isabs(path):
                            path = os.path.join(image_dir, path)
                        image_paths.append(path)
    
    elif ext == '.jsonl':
        with open(list_file, 'r') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # Try common field names
                    for field in ['image', 'path', 'file_name', 'filename']:
                        if field in item:
                            path = item[field]
                            if image_dir and not os.path.isabs(path):
                                path = os.path.join(image_dir, path)
                            image_paths.append(path)
                            break
    
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    return image_paths


def extract_vgg19_features(image_paths, batch_size=32, num_workers=4, 
                          device=None, show_progress=True):
    """Extract VGG-19 features from images.
    
    Args:
        image_paths: List of image file paths
        batch_size: Batch size for processing
        num_workers: Number of data loading workers
        device: Device to use (defaults to CUDA if available)
        show_progress: Whether to show progress bar
    
    Returns:
        numpy array of shape (n_images, 4096)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load VGG19 model
    print("Loading VGG-19 model...")
    vgg19_model = models.vgg19(pretrained=True)
    # Remove the last classification layer to get 4096D features
    vgg19_model.classifier = torch.nn.Sequential(
        *list(vgg19_model.classifier.children())[:-3]
    )
    vgg19_model.eval()
    vgg19_model.to(device)
    
    # Define transform (standard ImageNet preprocessing)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
    ])
    
    # Create dataset and dataloader
    dataset = ImageDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                           num_workers=num_workers, shuffle=False)
    
    # Extract features
    features = []
    indices = []
    
    iterator = tqdm(dataloader, desc="Extracting features") if show_progress else dataloader
    
    for batch_images, batch_indices in iterator:
        batch_images = batch_images.to(device)
        
        with torch.no_grad():
            batch_features = vgg19_model(batch_images)
            batch_features = batch_features.cpu().numpy()
            features.extend(batch_features)
            indices.extend(batch_indices.numpy())
    
    # Reorder features to match input order
    features = np.array(features)
    indices = np.array(indices)
    sorted_features = features[np.argsort(indices)]
    
    return sorted_features


def main():
    parser = argparse.ArgumentParser(
        description='Extract VGG-19 features from images'
    )
    parser.add_argument('input', help='Input file listing images (txt/json/jsonl)')
    parser.add_argument('output', help='Output .npy file for features')
    parser.add_argument('--image-dir', help='Base directory for image paths')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for processing (default: 32)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    parser.add_argument('--device', choices=['cuda', 'cpu'],
                       help='Device to use (default: auto-detect)')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bar')
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load image list
    print(f"Loading image list from {args.input}...")
    image_paths = load_image_list(args.input, args.image_dir)
    print(f"Found {len(image_paths)} images")
    
    # Verify images exist
    missing = [p for p in image_paths if not os.path.exists(p)]
    if missing:
        print(f"Warning: {len(missing)} images not found")
        if len(missing) <= 10:
            for m in missing:
                print(f"  - {m}")
        else:
            print(f"  (showing first 10)")
            for m in missing[:10]:
                print(f"  - {m}")
    
    # Extract features
    features = extract_vgg19_features(
        image_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        show_progress=not args.no_progress
    )
    
    # Save features
    print(f"Saving features to {args.output}...")
    np.save(args.output, features)
    print(f"Features shape: {features.shape}")
    print("Done!")


if __name__ == '__main__':
    main()