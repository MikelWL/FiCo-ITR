import sys
import os
import pytest
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from fico_itr.tasks import category_retrieval, instance_retrieval
from fico_itr.similarity import compute_similarity
from .test_utils import load_categories

# Fixtures for loading test data (unchanged)
@pytest.fixture
def cg_img_embs():
    return np.load(os.path.join(os.path.dirname(__file__), 'test_data', 'cg_img_embs_f30k.npy'))

@pytest.fixture
def cg_cap_embs():
    return np.load(os.path.join(os.path.dirname(__file__), 'test_data', 'cg_cap_embs_f30k.npy'))

@pytest.fixture
def fg_img_embs():
    return np.load(os.path.join(os.path.dirname(__file__), 'test_data', 'fg_img_embs_f30k.npy'))

@pytest.fixture
def fg_cap_embs():
    return np.load(os.path.join(os.path.dirname(__file__), 'test_data', 'fg_cap_embs_f30k.npy'))

@pytest.fixture
def category_labels():
    return load_categories('f30k')

# Test similarity computation
def test_compute_similarity(fg_img_embs, fg_cap_embs):
    similarity_matrix = compute_similarity(fg_img_embs, fg_cap_embs, measure='cosine')
    assert similarity_matrix.shape == (len(fg_img_embs), len(fg_cap_embs)), f"Expected shape {(len(fg_img_embs), len(fg_cap_embs))}, but got {similarity_matrix.shape}"
    assert np.all(similarity_matrix >= -1) and np.all(similarity_matrix <= 1), "Cosine similarity values should be between -1 and 1"
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Similarity matrix range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")

# Test category retrieval
def test_category_retrieval(cg_img_embs, cg_cap_embs, category_labels):
    similarity_matrix = compute_similarity(cg_img_embs, cg_cap_embs, measure='cosine')
    results = category_retrieval(similarity_matrix, category_labels)
    
    assert 'mAP' in results, "mAP not found in results"
    assert 0 <= results['mAP'] <= 1, f"mAP should be between 0 and 1, but got {results['mAP']}"
    print(f"Category Retrieval Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")

# Test instance retrieval
def test_instance_retrieval(fg_img_embs, fg_cap_embs):
    similarity_matrix = compute_similarity(fg_img_embs, fg_cap_embs, measure='cosine')
    i2t_results, t2i_results = instance_retrieval(similarity_matrix)
    
    for results in [i2t_results, t2i_results]:
        assert all(metric in results for metric in ['R@1', 'R@5', 'R@10', 'MedianR', 'MeanR']), "Missing metrics in results"
        assert all(0 <= results[f'R@{k}'] <= 100 for k in [1, 5, 10]), f"R@k should be percentages, but got {results}"
    
    print(f"Image-to-Text Retrieval Results:")
    for key, value in i2t_results.items():
        print(f"  {key}: {value:.2f}")
    
    print(f"Text-to-Image Retrieval Results:")
    for key, value in t2i_results.items():
        print(f"  {key}: {value:.2f}")

# Test different similarity measures
@pytest.mark.parametrize("measure", ['cosine', 'euclidean', 'inner_product'])
def test_similarity_measures(fg_img_embs, fg_cap_embs, measure):
    similarity_matrix = compute_similarity(fg_img_embs, fg_cap_embs, measure=measure)
    assert similarity_matrix.shape == (len(fg_img_embs), len(fg_cap_embs)), f"Expected shape {(len(fg_img_embs), len(fg_cap_embs))}, but got {similarity_matrix.shape}"
    print(f"{measure.capitalize()} similarity matrix shape: {similarity_matrix.shape}")
    print(f"{measure.capitalize()} similarity matrix range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")

# Test error handling
def test_invalid_similarity_measure():
    with pytest.raises(ValueError, match="Invalid similarity measure: invalid_measure"):
        compute_similarity(np.array([[1, 2]]), np.array([[3, 4]]), measure='invalid_measure')

def test_mismatched_dimensions():
    with pytest.raises(ValueError):
        compute_similarity(np.random.rand(10, 5), np.random.rand(10, 6))