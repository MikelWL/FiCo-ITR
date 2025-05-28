# Query2Label Integration for FiCo-ITR

This directory contains integration scripts for using Query2Label (Q2L) to generate semantic category labels for datasets that lack native multi-label annotations, such as Flickr30K.

## Overview

Our integration provides:
- `flickr30k_generate_labels.py`: Script to generate COCO-compatible labels for Flickr30K
- `flickrdataset.py`: Dataset loader compatible with Q2L's framework
- Direct NumPy output format compatible with FiCo-ITR

## Prerequisites

1. **Setup Query2Label**: Follow the installation instructions at [Query2Label repository](https://github.com/SlongLiu/query2label)
2. **Download pretrained models**: Obtain Q2L pretrained checkpoints from their repository

## Integration Steps

### 1. Copy Integration Files

Copy our integration files to the Query2Label repository:

```bash
# Copy the main script to Q2L root directory
cp flickr30k_generate_labels.py /path/to/query2label/

# Copy the dataset loader to Q2L's dataset directory
cp flickrdataset.py /path/to/query2label/dataset/
```

### 2. Prepare Your Dataset

Ensure your Flickr30K images are in a single directory:

```
/path/to/flickr30k_images/
├── 1000092795.jpg
├── 10002456.jpg
└── ...
```

### 3. Run Label Generation

From the Query2Label directory, run:

```bash
python flickr30k_generate_labels.py \
    --dataset_dir /path/to/flickr30k_images \
    --resume /path/to/q2l_checkpoint.pth \
    --arch Q2L-R101-448 \
    --output_labels flickr30k_labels.npy \
    --output_names flickr30k_names.txt
```

## Usage Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset_dir` | Path to Flickr30K images directory | Required |
| `--resume` | Path to Q2L pretrained checkpoint | Required |
| `--arch` | Model architecture (must match checkpoint) | Q2L-R101-448 |
| `--output_labels` | Output path for labels NumPy file | flickr30k_labels.npy |
| `--output_names` | Output path for image names text file | flickr30k_names.txt |
| `--batch-size` | Batch size for inference | 16 |
| `--img_size` | Input image size | 448 |
| `--threshold` | Threshold for positive label assignment | 0.5 |

## Output Format

The script generates:
- **Labels file** (`.npy`): Binary matrix of shape `(n_images, 80)` where each row represents an image and columns represent COCO categories
- **Names file** (`.txt`): Image filenames in the same order as the labels (one per line)

## Example Usage with FiCo-ITR

Once you have generated the labels, you can use them with FiCo-ITR:

```python
import numpy as np
from fico_itr import RetrievalTask

# Load generated labels
labels = np.load('flickr30k_labels.npy')

# Use for category-level retrieval evaluation
task = RetrievalTask(
    labels=labels,
    retrieval_type='category'
)
```

## Notes

- Requires GPU for efficient processing
- Uses COCO's 80-category taxonomy
- Images are processed in the order returned by `os.listdir()`