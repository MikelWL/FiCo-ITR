import os
import numpy as np
import json
from pathlib import Path
from fico_itr.similarity import compute_similarity
from fico_itr.tasks import category_retrieval, instance_retrieval

def load_data(model, dataset, modality=None):
    """Load data for the given model and dataset."""
    base_path = Path("results_data")
    
    if modality:
        # For models with separate embeddings
        file_path = base_path / f"{model}_{dataset}_{modality}.npy"
    else:
        # For models with pre-computed similarity matrices
        file_path = base_path / f"{model}_{dataset}_sim.npy"
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return np.load(file_path)

def load_labels(dataset):
    """Load category labels from .npy file."""
    if dataset == 'f30k':
        file_name = "flickr30k-karpathy-test-labels.npy"
    elif dataset == 'coco':
        file_name = "coco-karpathy-testall-labels.npy"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    file_path = Path("results_data") / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Label file not found: {file_path}")

    return np.load(file_path)

def is_precomputed_similarity(model):
    """Determine if the model uses pre-computed similarity matrices."""
    return model in ['blip2', 'xvlm', 'imram', 'scan'] or model.endswith('(nf)')

def run_demo(model, dataset, labels):
    print(f"Running demo for {model} on {dataset}")

    # Determine caption distribution for this dataset
    if dataset == 'coco' and model in ['beit3', 'xvlm(nf)', 'xvlm', 'blip2']:
        # Load caption-to-image indices for models with 25010 captions
        captions_per_image = np.load('./results_data/mscoco_test_indices.npy').tolist()
    elif dataset == 'coco':
        # COCO standard: most models use 25000 captions (5 per image)
        captions_per_image = 5
    elif dataset == 'f30k':
        # Flickr30k has exactly 5 captions per image
        captions_per_image = 5
    else:
        # Auto-detect for unknown datasets
        captions_per_image = 'auto'

    if is_precomputed_similarity(model):
        # Load pre-computed similarity matrix
        if model in ['blip2', 'xvlm']:
            # Models with task-specific matrices
            sim_i2t = load_data(model, dataset, 'sim_i2t')
            sim_t2i = load_data(model, dataset, 'sim_t2i')
        else:
            # Models with a single similarity matrix
            similarity_matrix = load_data(model, dataset)
    else:
        # Load separate embeddings and compute similarity
        img_embs = load_data(model, dataset, 'img')
        txt_embs = load_data(model, dataset, 'txt')
        similarity_matrix = compute_similarity(img_embs, txt_embs, measure='cosine')
        
        # Note: Square matrices (vsrn/ucch) still use original caption ratios
        # to preserve ground truth relationships

    print("Performing instance-level retrieval...")
    if model in ['blip2', 'xvlm']: # Models with task-specific matrices
        i2t_instance_results, t2i_instance_results = instance_retrieval(
            sim_i2t, sim_t2i, captions_per_image=captions_per_image
        )
    else:
        i2t_instance_results, t2i_instance_results = instance_retrieval(
            similarity_matrix, captions_per_image=captions_per_image
        )

    print("Instance-level Retrieval Results:")
    print("Image-to-Text:")
    for metric, value in i2t_instance_results.items():
        print(f"  {metric}: {value:.2f}")
    print("Text-to-Image:")
    for metric, value in t2i_instance_results.items():
        print(f"  {metric}: {value:.2f}")

    print("Performing category-level retrieval...")
    if model in ['blip2', 'xvlm']: # Models with task-specific matrices
        i2t_catategory_results, t2i_catategory_results = category_retrieval(
            sim_i2t, labels, sim2=sim_t2i, captions_per_image=captions_per_image
        )
    else:
        i2t_catategory_results, t2i_catategory_results = category_retrieval(
            similarity_matrix, labels, captions_per_image=captions_per_image
        )
    
    print("Category-level Retrieval Results:")
    print(f"Image-to-Text:{i2t_catategory_results}")
    print(f"Text-to-Image:{t2i_catategory_results}")

    print("\n")


if __name__ == "__main__":
    models = ["beit3", "blip2", "vsrn", "xvlm", "ucch", "dadh", "adv(2048bit)", "adv(64bit)", "imram", "scan", "blip2(nf)", "xvlm(nf)"]
    datasets = ["f30k", "coco"]

    for dataset in datasets:
        print(f"Loading labels for {dataset}...")
        try:
            labels = load_labels(dataset)
        except FileNotFoundError as e:
            print(f"Error loading labels for {dataset}: {str(e)}")
            continue

        for model in models:
            try:
                run_demo(model, dataset, labels)
            except FileNotFoundError as e:
                print(f"Skipping {model} on {dataset}: {str(e)}")
            except Exception as e:
                print(f"Error running demo for {model} on {dataset}: {str(e)}")
    
    