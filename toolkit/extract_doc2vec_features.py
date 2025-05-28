#!/usr/bin/env python3
"""
Extract 300D Doc2Vec features for text in coarse-grained models.

This tool generates Doc2Vec embeddings for text captions, commonly used in
coarse-grained retrieval models. It can either train a new model or use an
existing one.
"""

import argparse
import json
import os
import random
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm

# Download punkt tokenizer if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')


def preprocess_text(text):
    """Preprocess the text: tokenize and lowercase."""
    return word_tokenize(text.lower())


def extract_doc2vec_features(caption_texts, model):
    """Extracts doc2vec features for a list of texts."""
    features = []
    for caption in tqdm(caption_texts, desc="Extracting features"):
        words = preprocess_text(caption)
        vector = model.infer_vector(words)
        features.append(vector)
    return np.array(features)


def load_captions(caption_file, format='json'):
    """Load captions from file.
    
    Supports multiple formats:
    - json: {'image_id': ['caption1', 'caption2', ...], ...}
    - txt: One caption per line
    - jsonl: One JSON object per line with 'caption' field
    """
    captions = []
    
    if format == 'json':
        with open(caption_file, 'r') as f:
            data = json.load(f)
            # Handle different JSON structures
            if isinstance(data, dict):
                # Assume dict of image_id: [captions]
                for img_id, caps in data.items():
                    if isinstance(caps, list):
                        captions.extend(caps)
                    else:
                        captions.append(caps)
            elif isinstance(data, list):
                # Assume list of caption objects
                for item in data:
                    if isinstance(item, str):
                        captions.append(item)
                    elif isinstance(item, dict) and 'caption' in item:
                        captions.append(item['caption'])
    
    elif format == 'txt':
        with open(caption_file, 'r') as f:
            captions = [line.strip() for line in f if line.strip()]
    
    elif format == 'jsonl':
        with open(caption_file, 'r') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if 'caption' in item:
                        captions.append(item['caption'])
    
    return captions


def train_doc2vec_model(captions, vector_size=300, window=5, min_count=1, 
                       epochs=10, workers=4):
    """Train a new Doc2Vec model on the provided captions."""
    print(f"Training Doc2Vec model on {len(captions)} captions...")
    
    # Prepare tagged documents
    tagged_documents = []
    for idx, caption in enumerate(captions):
        words = preprocess_text(caption)
        tagged_document = TaggedDocument(words=words, tags=[idx])
        tagged_documents.append(tagged_document)
    
    # Shuffle documents
    random.shuffle(tagged_documents)
    
    # Initialize model
    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )
    
    # Build vocabulary
    model.build_vocab(tagged_documents)
    
    # Train the model
    for epoch in range(epochs):
        print(f"Training epoch {epoch+1}/{epochs}...")
        random.shuffle(tagged_documents)
        model.train(tagged_documents, 
                   total_examples=model.corpus_count, 
                   epochs=model.epochs)
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Extract Doc2Vec features for text captions'
    )
    parser.add_argument('input', help='Input caption file')
    parser.add_argument('output', help='Output .npy file for features')
    parser.add_argument('--format', choices=['json', 'txt', 'jsonl'], 
                       default='json',
                       help='Input file format (default: json)')
    parser.add_argument('--model', help='Path to existing Doc2Vec model')
    parser.add_argument('--save-model', help='Path to save trained model')
    parser.add_argument('--train-captions', 
                       help='Additional captions for training (if training new model)')
    
    # Training parameters
    parser.add_argument('--vector-size', type=int, default=300,
                       help='Dimensionality of feature vectors (default: 300)')
    parser.add_argument('--window', type=int, default=5,
                       help='Maximum distance between current and predicted word (default: 5)')
    parser.add_argument('--min-count', type=int, default=1,
                       help='Ignore words with frequency lower than this (default: 1)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of worker threads (default: 4)')
    
    args = parser.parse_args()
    
    # Load captions to process
    print(f"Loading captions from {args.input}...")
    captions = load_captions(args.input, args.format)
    print(f"Loaded {len(captions)} captions")
    
    # Load or train model
    if args.model:
        print(f"Loading existing model from {args.model}...")
        model = Doc2Vec.load(args.model)
    else:
        # Load training captions
        train_captions = captions.copy()
        if args.train_captions:
            print(f"Loading additional training captions from {args.train_captions}...")
            additional_captions = load_captions(args.train_captions, args.format)
            train_captions.extend(additional_captions)
            print(f"Total training captions: {len(train_captions)}")
        
        # Train new model
        model = train_doc2vec_model(
            train_captions,
            vector_size=args.vector_size,
            window=args.window,
            min_count=args.min_count,
            epochs=args.epochs,
            workers=args.workers
        )
        
        # Save model if requested
        if args.save_model:
            print(f"Saving model to {args.save_model}...")
            model.save(args.save_model)
    
    # Extract features
    print("Extracting Doc2Vec features...")
    features = extract_doc2vec_features(captions, model)
    
    # Save features
    print(f"Saving features to {args.output}...")
    np.save(args.output, features)
    print(f"Features shape: {features.shape}")
    print("Done!")


if __name__ == '__main__':
    main()