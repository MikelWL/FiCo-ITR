
# Image-Text Alignment in FiCo-ITR

## Overview

FiCo-ITR (Fine-grained and Coarse-grained Image-Text Retrieval) handles various alignment scenarios between images and captions. This document explains the alignment mechanism, assumptions, and considerations for using the library effectively.

## Alignment Scenarios

1. Balanced Sets: Equal number of image and caption embeddings (e.g., 5000 images, 5000 captions)
2. Unbalanced Sets: Unequal number of image and caption embeddings (e.g., 1000 images, 5000 captions)
3. Pre-aligned Sets: Embeddings where images have been repeated to match captions

## Key Assumptions

1. Labels correspond to unique images, not duplicated embeddings.
2. The number of labels is always equal to or less than the number of image and caption embeddings.
3. The ratio between embeddings and labels is consistent throughout the dataset.
4. Alignment ratios are assumed to be small integers (e.g., 1:1, 1:5, 5:1).

## How Alignment Works

1. Alignment is handled internally within the `category_retrieval` function.
2. The function detects alignment based on the shapes of the similarity matrix and the provided labels.
3. Alignment ratios are calculated as:
   - `img_ratio = n_images // n_labels`
   - `cap_ratio = n_captions // n_labels`
4. Labels are repeated to match the number of embeddings during category retrieval tasks.

## Important Considerations

1. Label Preparation: Provide labels corresponding to unique images, not duplicated embeddings.
2. Non-integer Ratios: Currently not supported. Ensure your data has clean integer ratios.

## Limitations and Future Work

1. Support for non-integer ratios and custom index mapping.
2. Optimization for very large datasets.
3. Handling of more complex alignment scenarios.
