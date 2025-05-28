import numpy as np
from fico_itr import compute_similarity, category_retrieval, instance_retrieval

# Load your image and text embeddings. Alternatively, directly use those produced by model
image_embeddings = np.load('results_data/vsrn_f30k_img.npy')
text_embeddings = np.load('results_data/vsrn_f30k_txt.npy')
labels = np.load('results_data/flickr30k-karpathy-test-labels.npy')

# Compute similarity matrix
similarity_matrix = compute_similarity(image_embeddings, text_embeddings, measure='cosine')

# Perform image-to-text retrieval
# Note: vsrn uses square matrices (5000×5000) from duplicating 1000 images 5 times
# We need to specify captions_per_image=5 to handle this correctly
i2t_instance_results, t2i_instance_results = instance_retrieval(similarity_matrix, captions_per_image=5)
i2t_category_results, t2i_category_results = category_retrieval(similarity_matrix, labels, captions_per_image=5)

print(f"Instance Retrieval Results: \n Image-to-Text: {i2t_instance_results} \n Text-to-Image: {t2i_instance_results}")
print(f"Category Retrieval Results: \n Image-to-Text: {i2t_category_results} \n Text-to-Image: {t2i_category_results}")