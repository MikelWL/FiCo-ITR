import numpy as np
from fico_itr import compute_similarity, category_retrieval, instance_retrieval
from scipy import io

# Load your image and text embeddings. Alternatively, directly use those produced by model
image_embeddings = np.load('results_data/vsrn_f30k_img.npy')
text_embeddings = np.load('results_data/vsrn_f30k_txt.npy')
mat_data = io.loadmat('results_data/flickr30k-karpathy-test-lall.mat')
labels = mat_data['LAll']

# Compute similarity matrix
similarity_matrix = compute_similarity(image_embeddings, text_embeddings, measure='cosine')

# Perform image-to-text retrieval
i2t_instance_results, t2i_instance_results = instance_retrieval(similarity_matrix)
i2t_category_results, t2i_category_results = category_retrieval(similarity_matrix, labels)

print(f"Instance Retrieval Results: \n Image-to-Text: {i2t_instance_results} \n Text-to-Image: {t2i_instance_results}")
print(f"Category Retrieval Results: \n Image-to-Text: {i2t_category_results} \n Text-to-Image: {t2i_category_results}")