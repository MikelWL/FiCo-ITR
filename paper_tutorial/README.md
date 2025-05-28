# FiCo-ITR Paper Tutorial

This tutorial demonstrates how to reproduce the results from the FiCo-ITR paper using pre-computed embeddings and similarity matrices.

## Overview

This directory contains:
- `demo.py` - Full demo that evaluates all models from the paper
- `demo_simplified.py` - Minimal example showing basic usage
- `results_data/` - Pre-computed embeddings and similarity matrices

## Setup

### 1. Install Dependencies

```bash
pip install numpy fico_itr
```

### 2. Download Pre-computed Data

Download the pre-computed embeddings, similarity matrices, labels and image-caption mappings:

**[📥 Download from Google Drive](https://drive.google.com/drive/folders/1r2allfSdV1K8s-e4tukjLQA42zlEK46j?usp=drive_link)**

Extract all `.npy` files into the `results_data/` directory.

### 3. Data Files Structure

```
paper_tutorial/
├── demo.py
├── demo_simplified.py
└── results_data/
    ├── mscoco_test_indices.npy      # Caption mapping for COCO
    ├── *_f30k_*.npy                  # Flickr30k embeddings/similarities
    ├── *_coco_*.npy                  # MS-COCO embeddings/similarities
    └── *.npy                         # Labels for category evaluation
```

## Running the Demos

### Full Evaluation (All Models)

```bash
python demo.py
```

This evaluates all models from the paper on both Flickr30k and MS-COCO datasets, showing:
- Instance-level retrieval metrics (R@1, R@5, R@10, MedianR, MeanR)
- Category-level retrieval metrics (mAP)

## Expected Output

```
Running demo for beit3 on f30k
Performing instance-level retrieval...
Instance-level Retrieval Results:
Image-to-Text:
  R@1: 96.30
  R@5: 99.70
  R@10: 100.00
  MedianR: 1.00
  MeanR: 1.08
Text-to-Image:
  R@1: 86.16
  R@5: 97.68
  R@10: 98.82
  MedianR: 1.00
  MeanR: 1.80
Performing category-level retrieval...
Category-level Retrieval Results:
Image-to-Text:0.9442431761039228
Text-to-Image:0.9483986751278153

(...)
```

### Simplified Example

```bash
python demo_simplified.py
```

This shows a minimal example using VSRN on Flickr30k.


## Models Included

The demo includes results for:

| Model | Datasets | Special Handling |
|-------|----------|------------------|
| BEiT-3 | F30k, COCO | COCO uses caption mapping |
| SCAN | F30k, COCO | Standard format |
| VSRN | F30k, COCO | Square matrices (5000×5000) |
| UCCH | F30k, COCO | Square matrices |
| DADH | F30k, COCO | Transposed matrices |
| ADV | F30k, COCO | Binary embeddings |
| IMRAM | F30k, COCO | Pre-computed similarities |
| BLIP-2 | F30k, COCO | Separate i2t/t2i matrices |
| X-VLM | F30k, COCO | Separate i2t/t2i matrices |


## Notes

## Citation

```bibtex
@article{williams-lekuona2025ficoitr,
  author  = {Mikel Williams-Lekuona and Georgina Cosma},
  title   = {FiCo-ITR: Bridging Fine-Grained and Coarse-Grained Image-Text Retrieval for Comparative Performance Analysis},
  journal = {International Journal of Multimedia Information Retrieval},
  volume  = {14},
  number  = {2},
  pages   = {20},
  year    = {2025},
  publisher={Springer}
}
```