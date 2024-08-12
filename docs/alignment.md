
# Image-Text Alignment in FiCo-ITR

## Overview

FiCo-ITR (Fine-grained and Coarse-grained Image-Text Retrieval) is designed to handle various alignment scenarios between images and captions. This document explains the alignment mechanism, assumptions, and considerations for using the library effectively.

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

1. The `ImageTextRetrieval` class detects alignment based on the shapes of image embeddings, caption embeddings, and labels.
2. Alignment ratios are calculated as:
   - `img_ratio = len(image_embeddings) // len(labels)`
   - `cap_ratio = len(caption_embeddings) // len(labels)`
3. Labels are repeated to match the number of embeddings during category retrieval tasks.

## Code Operation

1. Initialization:
   ```python
   retriever = ImageTextRetrieval(image_embeddings, caption_embeddings, labels)
   ```

2. Alignment Detection:
   Alignment ratios are calculated automatically when the `ImageTextRetrieval` object is initialized with labels. If labels are not provided during initialization, the ratios will be calculated when `prepare_for_category_retrieval()` is called.

   You can check the current alignment information at any time using the `get_alignment_info()` method:

   ```python
   alignment_info = retriever.get_alignment_info()
   print(alignment_info)
   ```

   This will return a dictionary containing the number of image embeddings, caption embeddings, labels (if provided), and the calculated ratios (if labels were provided or if `prepare_for_category_retrieval()` has been called).

3. Label Alignment:
   ```python
   aligned_labels = np.repeat(self.labels, self.img_ratio, axis=0)
   ```

4. Similarity Computation:
   - Uses original embeddings without modification
   - Alignment is considered when interpreting results

## Important Considerations

1. Label Preparation: Provide labels corresponding to unique images, not duplicated embeddings.
2. Non-integer Ratios: Currently not supported. Ensure your data has clean integer ratios.
3. Memory Usage: Be aware of potential memory issues with large datasets, especially for Euclidean distance calculations.
4. Performance: Alignment detection and label repetition may impact performance for very large datasets.

## Limitations and Future Work

1. Support for non-integer ratios and custom index mapping.
2. Optimization for very large datasets.
3. Handling of more complex alignment scenarios.

## Best Practices

1. Always verify the detected alignment ratios before proceeding with retrieval tasks.
2. Use the `get_alignment_info()` method to inspect alignment details.
3. Ensure consistent data preparation across your dataset.

## Examples

See `examples/alignment_example.py` for detailed usage examples demonstrating various alignment scenarios.
