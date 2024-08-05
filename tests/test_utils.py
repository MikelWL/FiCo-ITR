import os
import numpy as np
import h5py

def load_categories(dataset):
    """
    Load category labels for the specified dataset.
    Args:
        dataset (str): Name of the dataset. Must be 'f30k' for Flickr30k.
    Returns:
        np.ndarray: Category labels array with shape (num_samples, num_categories).
    """
    if dataset != 'f30k':
        raise ValueError("Invalid dataset name. Currently only 'f30k' is supported.")

    path = os.path.join(os.path.dirname(__file__), 'test_data', 'flickr30k-karpathy-test-lall.mat')
    
    with h5py.File(path, 'r') as f:
        labels = f['LAll'][:]
    
    return labels.T  # Transpose to get (num_samples, num_categories)
