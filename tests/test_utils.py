import os
import numpy as np
import hdf5storage

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

    try:
        mat_data = hdf5storage.loadmat(path)
    except IOError as e:
        raise IOError(f"Unable to read file at {path}. Error: {str(e)}")
    except KeyError:
        raise KeyError("Expected key 'LAll' not found in .mat file")
    
    labels = mat_data['LAll']
    
    return labels  # Transpose to get (num_samples, num_categories)
