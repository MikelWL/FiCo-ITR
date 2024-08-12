import sys
import os
import pytest
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from fico_itr.tasks import ImageTextRetrieval, category_retrieval, instance_retrieval_i2t, instance_retrieval_t2i
from fico_itr.similarity import compute_similarity
from .test_utils import load_categories

# Fixtures for loading test data
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
    retriever = ImageTextRetrieval(fg_img_embs, fg_cap_embs)
    similarity_matrix = retriever.compute_similarity(measure='cosine')
    assert similarity_matrix.shape == (len(retriever.img_indices), len(retriever.cap_indices)), f"Expected shape {(len(retriever.img_indices), len(retriever.cap_indices))}, but got {similarity_matrix.shape}"
    assert np.all(similarity_matrix >= -1) and np.all(similarity_matrix <= 1), "Cosine similarity values should be between -1 and 1"
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Similarity matrix range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")

# Test category retrieval
def test_category_retrieval(cg_img_embs, cg_cap_embs, category_labels):
    retriever = ImageTextRetrieval(cg_img_embs, cg_cap_embs, category_labels)
    
    results = category_retrieval(retriever)
    
    alignment_info = retriever.get_alignment_info()
    print(f"Alignment info: {alignment_info}")

    assert 'mAP' in results, "mAP not found in results"
    assert 0 <= results['mAP'] <= 1, f"mAP should be between 0 and 1, but got {results['mAP']}"
    print(f"Category Retrieval Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")

# Test image-to-text retrieval
def test_image_to_text_retrieval(fg_img_embs, fg_cap_embs):
    retriever = ImageTextRetrieval(fg_img_embs, fg_cap_embs)
    results = instance_retrieval_i2t(retriever)
    
    assert len(results) == 5, f"Expected 5 results, but got {len(results)}"
    assert all(0 <= r <= 100 for r in results[:3]), f"R@1, R@5, R@10 should be percentages, but got {results[:3]}"
    print(f"Image-to-Text Retrieval Results:")
    print(f"  R@1: {results[0]:.2f}%, R@5: {results[1]:.2f}%, R@10: {results[2]:.2f}%")
    print(f"  Median rank: {results[3]:.2f}, Mean rank: {results[4]:.2f}")

def test_text_to_image_retrieval(fg_img_embs, fg_cap_embs):
    retriever = ImageTextRetrieval(fg_img_embs, fg_cap_embs)
    results = instance_retrieval_t2i(retriever)
    
    assert len(results) == 5, f"Expected 5 results, but got {len(results)}"
    assert all(0 <= r <= 100 for r in results[:3]), f"R@1, R@5, R@10 should be percentages, but got {results[:3]}"
    print(f"Text-to-Image Retrieval Results:")
    print(f"  R@1: {results[0]:.2f}%, R@5: {results[1]:.2f}%, R@10: {results[2]:.2f}%")
    print(f"  Median rank: {results[3]:.2f}, Mean rank: {results[4]:.2f}")

# Test different similarity measures
@pytest.mark.parametrize("measure", ['cosine', 'euclidean', 'inner_product'])
def test_similarity_measures(fg_img_embs, fg_cap_embs, measure):
    retriever = ImageTextRetrieval(fg_img_embs, fg_cap_embs)
    similarity_matrix = retriever.compute_similarity(measure=measure)
    assert similarity_matrix.shape == (len(retriever.img_indices), len(retriever.cap_indices)), f"Expected shape {(len(retriever.img_indices), len(retriever.cap_indices))}, but got {similarity_matrix.shape}"
    print(f"{measure.capitalize()} similarity matrix shape: {similarity_matrix.shape}")
    print(f"{measure.capitalize()} similarity matrix range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")

# Test error handling
def test_invalid_similarity_measure():
    retriever = ImageTextRetrieval(np.array([[1, 2]]), np.array([[3, 4]]))
    with pytest.raises(ValueError, match="Invalid similarity measure: invalid_measure"):
        retriever.compute_similarity(measure='invalid_measure')

def test_mismatched_dimensions():
    with pytest.raises(ValueError, match="Labels are required for category retrieval tasks."):
        retriever = ImageTextRetrieval(np.random.rand(10, 5), np.random.rand(10, 5))
        category_retrieval(retriever)