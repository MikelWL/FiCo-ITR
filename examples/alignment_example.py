import numpy as np
from fico_itr import ImageTextRetrieval, category_retrieval

# Balanced set example (1:1 ratio)
img_emb_balanced = np.random.rand(1000, 512)
cap_emb_balanced = np.random.rand(1000, 512)
labels_balanced = np.random.randint(0, 2, (1000, 10))

retriever_balanced = ImageTextRetrieval(img_emb_balanced, cap_emb_balanced, labels_balanced)
print("Balanced set alignment info:", retriever_balanced.get_alignment_info())

# Unbalanced set example (1:5 ratio)
img_emb_unbalanced = np.random.rand(1000, 512)
cap_emb_unbalanced = np.random.rand(5000, 512)
labels_unbalanced = np.random.randint(0, 2, (1000, 10))

retriever_unbalanced = ImageTextRetrieval(img_emb_unbalanced, cap_emb_unbalanced, labels_unbalanced)
print("Unbalanced set alignment info:", retriever_unbalanced.get_alignment_info())

# Category retrieval example
results = category_retrieval(retriever_unbalanced)
print("Alignment info after category retrieval:", retriever_unbalanced.get_alignment_info())
print("Category retrieval results:", results)

# Example without labels
retriever_no_labels = ImageTextRetrieval(img_emb_balanced, cap_emb_balanced)
print("Alignment info without labels:", retriever_no_labels.get_alignment_info())