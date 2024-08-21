import os
import numpy as np
import json
import hdf5storage
from pathlib import Path
import torch
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
    """Load category labels from .mat file."""
    if dataset == 'f30k':
        file_name = "flickr30k-karpathy-test-lall.mat"
    elif dataset == 'coco':
        file_name = "coco-karpathy-testall-lall.mat"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    file_path = Path("results_data") / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Label file not found: {file_path}")

    mat_data = hdf5storage.loadmat(str(file_path))
    return mat_data['LAll']

def is_precomputed_similarity(model):
    """Determine if the model uses pre-computed similarity matrices."""
    return model in ['blip2', 'xvlm', 'imram', 'scan'] or model.endswith('(nf)')

def run_demo(model, dataset, labels):
    print(f"Running demo for {model} on {dataset}")

    if is_precomputed_similarity(model):
        # Load pre-computed similarity matrix
        if model in ['blip2', 'xvlm']:
            # Models with task-specific matrices
            sim_i2t = load_data(model, dataset, 'sim_i2t')
            sim_t2i = load_data(model, dataset, 'sim_t2i')
            similarity_matrix = (sim_i2t + sim_t2i.T) / 2  # Average for category retrieval
        else:
            # Models with a single similarity matrix
            similarity_matrix = load_data(model, dataset)
    else:
        # Load separate embeddings and compute similarity
        img_embs = load_data(model, dataset, 'img')
        txt_embs = load_data(model, dataset, 'txt')
        print("Computing similarity matrix...")
        similarity_matrix = compute_similarity(img_embs, txt_embs, measure='cosine')

    print("Performing instance-level retrieval...")
    if model in ['blip2', 'xvlm']:
        i2t_results = instance_retrieval(sim_i2t, direction='i2t')
        t2i_results = instance_retrieval(sim_t2i, direction='t2i')
    else:
        i2t_results, t2i_results = instance_retrieval(similarity_matrix)

    print("Instance-level Retrieval Results:")
    print("Image-to-Text:")
    for metric, value in i2t_results.items():
        print(f"  {metric}: {value:.2f}")
    print("Text-to-Image:")
    for metric, value in t2i_results.items():
        print(f"  {metric}: {value:.2f}")

    print("Performing category-level retrieval...")
    cat_results = category_retrieval(similarity_matrix, labels)
    
    print("Category-level Retrieval Results:")
    for metric, value in cat_results.items():
        print(f"  {metric}: {value:.4f}")

    print("\n")

def debug_run():
    models = ["dadh", "vsrn", "beit3"]
    # datasets = ["f30k", "coco"]
    datasets = ["f30k"]
    results = {}

    for dataset in datasets:
        print(f"Loading labels for {dataset}...")
        try:
            labels = load_labels(dataset)
        except FileNotFoundError as e:
            print(f"Error loading labels for {dataset}: {str(e)}")
            continue

        for model in models:
            try:
                print(f"Running demo for {model} on {dataset}")
                if is_precomputed_similarity(model):
                    # Load pre-computed similarity matrix
                    similarity_matrix = load_data(model, dataset)
                else:
                    # Load separate embeddings and compute similarity
                    img_embs = load_data(model, dataset, 'img')
                    txt_embs = load_data(model, dataset, 'txt')
                    print("Computing similarity matrix...")
                    similarity_matrix = compute_similarity(img_embs, txt_embs, measure='cosine')

                print("Performing instance-level retrieval...")
                i2t_results, t2i_results = instance_retrieval(similarity_matrix)

                print("Performing category-level retrieval...")
                cat_results = category_retrieval(similarity_matrix, labels)

                results.setdefault(model, {})[dataset] = {
                    "instance_retrieval": {
                        "i2t": i2t_results,
                        "t2i": t2i_results
                    },
                    "category_retrieval": cat_results
                }

            except FileNotFoundError as e:
                print(f"Skipping {model} on {dataset}: {str(e)}")
            except Exception as e:
                print(f"Error running demo for {model} on {dataset}: {str(e)}")

    return results

def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] // 5
    index_list = []

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], 5 * (index + bs))
                im2 = images[5 * index:mx:5]
                d2 = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions).cuda())
                d2 = d2.cpu().numpy()
            d = d2[index % bs]
        else:
            d = np.dot(im, captions.T).flatten()
        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp

        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] // 5
    ims = np.array([images[i] for i in range(0, len(images), 5)])

    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        # Compute scores
        if measure == 'order':
            bs = 100
            if 5 * index % bs == 0:
                mx = min(captions.shape[0], 5 * index + bs)
                q2 = captions[5 * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                               torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (5 * index) % bs:(5 * index) % bs + 5].T
        else:
            d = np.dot(queries, ims.T)
        inds = np.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = np.argsort(d[i])[::-1]
            ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score

if __name__ == "__main__":

    '''
    img_embs = load_data('vsrn', 'f30k', 'img')
    txt_embs = load_data('vsrn', 'f30k', 'txt')

    # original evaluation
    r, rt = i2t(img_embs, txt_embs, measure='cosine', return_ranks=True)
    ri, rti = t2i(img_embs, txt_embs, measure='cosine', return_ranks=True)

    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    print("rsum: %.1f" % rsum)
    print("Average i2t Recall: %.1f" % ar)
    print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
    print("Average t2i Recall: %.1f" % ari)
    print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    '''

    debug_results = debug_run()
    print(json.dumps(debug_results, indent=2))

    '''
    # Commented out for now as we are on debugging mode
    models = ["beit3", "blip2", "vsrn", "xvlm", "ucch", "dadh", "adv(2048bit)", "adv(64bit)", "imram", "scan", "blip2(nf)", "xvlm(nf)"]
    datasets = ["f30k", "coco"]
    datasets = ["f30k"]

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
    '''
    