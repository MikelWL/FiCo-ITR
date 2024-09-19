# FiCo-ITR Demo

This demo serves as an example usage of the FiCo-ITR (Fine-grained and Coarse-grained Image-Text Retrieval) library and provides a quick way to reproduce results using pre-computed embeddings.

## Purpose

1. Demonstrate how to use the FiCo-ITR library for evaluating image-text retrieval models.
2. Provide a benchmark for reproducing results with minimal setup.
3. Compare the FiCo-ITR implementation with original model evaluation code.

## Requirements

- Python 3.7+
- NumPy
- h5py
- FiCo-ITR library

## Setup

1. Install the required dependencies:

   ```
   pip install numpy h5py fico_itr
   ```

2. Download the pre-computed embeddings:

   [Download pre-computed embeddings](https://drive.google.com/drive/folders/1r2allfSdV1K8s-e4tukjLQA42zlEK46j?usp=drive_link)

   Extract the downloaded files into the `results_data` directory.

## Usage

Run the demo script:

```
python demo.py
```

This script will:

1. Load pre-computed embeddings and similarity matrices for the models which were evaluated in the paper for both Flickr30K and MS-COCO.
2. Use the FiCo-ITR library to compute similarity and perform retrieval tasks.

## Output

The demo will output results for:

1. Instance-level retrieval:
   - Image-to-text retrieval (R@1, R@5, R@10, MedianR, MeanR)
   - Text-to-image retrieval (R@1, R@5, R@10, MedianR, MeanR)
2. Category-level retrieval:
   - Mean Average Precision (mAP)

## Customization

You can modify the `demo.py` script to work with different models or datasets by changing the data loading functions and file paths.

## Known WIP issues towards full implemenation

 - Pre-computed asymmetric matrices not yet supported (e.g. XVLM(nf)). These cases are handled via exception and will be addressed in the full implementation.

## Citation

If you use this demo or the FiCo-ITR library in your research, please cite our paper:

```
[Citation details will be added upon paper acceptance]
```