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

   [Download pre-computed embeddings](INSERT_DOWNLOAD_LINK_HERE)

   Extract the downloaded files into the `results_data` directory.

## Usage

Run the demo script:

```
python demo.py
```

This script will:

1. Load pre-computed embeddings for the VSRN model on the Flickr30K dataset.
2. Run the original VSRN evaluation code to establish baseline results.
3. Use the FiCo-ITR library to compute similarity and perform retrieval tasks.
4. Display results for both methods, allowing for easy comparison.

## Output

The demo will output results for:

1. Instance-level retrieval:
   - Image-to-text retrieval (R@1, R@5, R@10, MedianR, MeanR)
   - Text-to-image retrieval (R@1, R@5, R@10, MedianR, MeanR)
2. Category-level retrieval:
   - Mean Average Precision (mAP)

Results will be displayed for both the original VSRN evaluation and the FiCo-ITR implementation.

## Customization

You can modify the `demo.py` script to work with different models or datasets by changing the data loading functions and file paths.

## Troubleshooting

If you encounter any issues or discrepancies in the results, please check:

1. Ensure you have downloaded the correct pre-computed embeddings.
2. Verify that the embeddings are placed in the correct directory.
3. Check that you're using the latest version of the FiCo-ITR library.

For further assistance, please open an issue on the FiCo-ITR GitHub repository.

## Citation

If you use this demo or the FiCo-ITR library in your research, please cite our paper:

```
[Citation details will be added upon paper acceptance]
```