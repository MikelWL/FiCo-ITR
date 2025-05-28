# FiCo-ITR Toolkit

This toolkit provides supplementary tools and best practice examples for image-text retrieval research. These are standalone utilities that complement the main FiCo-ITR evaluation library.

## Why a Separate Toolkit?

These tools are kept separate from the main library because:
- They demonstrate patterns rather than providing core functionality
- Most users will already have their preferred tools for these tasks
- They serve as educational examples of best practices

## Available Tools

### 1. Dataset Split Best Practices (`dataset_splits.py`)
**An educational example** showing the importance of reproducible evaluation:
- Demonstrates how to save and load split definitions
- Shows why split ordering matters for fair comparison
- Provides verification utilities to catch common mistakes

**Note**: For production use, we recommend established tools like `scikit-learn`'s `train_test_split` or PyTorch's data utilities. This script demonstrates the concepts.

### 2. VGG-19 Feature Extractor (`extract_vgg19_features.py`)
Extracts 4096-dimensional VGG-19 features for images (coarse-grained models):
- Standard ImageNet preprocessing
- Batch processing with GPU support
- Multiple input formats supported

### 3. Doc2Vec Feature Extractor (`extract_doc2vec_features.py`)
Generates 300-dimensional Doc2Vec features for text captions:
- Train new models or use existing ones
- Flexible input formats (JSON, TXT, JSONL)
- Direct NumPy output

### External Integrations

- **Bottom-Up Attention**: For region features, use the [original repository](https://github.com/peteanderson80/bottom-up-attention)
- **Query2Label**: For category labels, see our integration guide in `q2l_integration/`

## Best Practices Example

The `dataset_splits.py` script demonstrates key concepts:

```python
# Key lesson: Always save your splits for reproducibility!
from sklearn.model_selection import train_test_split

# Create splits
train_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=42)

# Save them!
with open('test_ids.txt', 'w') as f:
    for id in test_ids:
        f.write(f"{id}\n")

# Later, verify your features match the split
from dataset_splits import verify_split
verify_split(your_feature_ids, test_ids)  # Catches ordering mistakes!
```

## Where to Find Standard Splits

The standard Karpathy splits are simple text files with one image ID per line:
- **COCO**: [VSE++ repository](https://github.com/fartashf/vsepp/)
- **Flickr30K**: [VSE++ repository](https://github.com/fartashf/vsepp/)

## Usage Examples

### Feature Extraction
```bash
# VGG-19 features
python extract_vgg19_features.py images.txt vgg19_features.npy --image-dir /path/to/images

# Doc2Vec features  
python extract_doc2vec_features.py captions.json doc2vec_features.npy
```

### Understanding Split Management
```bash
# Run the educational demo
python dataset_splits.py

# This will show:
# - Why saving splits matters
# - How to verify split consistency
# - Best practices for reproducibility
```

## Requirements

```bash
# For the best practices demo
pip install numpy scikit-learn

# For feature extraction
pip install torch torchvision gensim nltk pillow
```
