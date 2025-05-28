#!/usr/bin/env python3
"""
Example of dataset split management best practices.

NOTE: This is an educational example showing patterns for reproducible evaluation.
For production use, we recommend established tools like:
- scikit-learn's train_test_split
- PyTorch's random_split
- TensorFlow's data splitting utilities

This script demonstrates:
1. How standard splits (like Karpathy) work
2. Importance of reproducibility in evaluation
3. Common pitfalls to avoid

This is intentionally kept as a standalone script in the toolkit rather than
part of the main library because most projects should use established tools.
"""

import os
import random
import numpy as np
from sklearn.model_selection import train_test_split


def load_split(split_file):
    """Load a split file containing one ID per line.
    
    This is the standard format used by Karpathy splits and most
    image-text retrieval benchmarks.
    
    Args:
        split_file: Path to text file with one ID per line
        
    Returns:
        List of IDs in the order they appear in the file
        
    Example:
        >>> test_ids = load_split('karpathy_test_ids.txt')
        >>> print(f"Test set has {len(test_ids)} samples")
    """
    ids = []
    with open(split_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                ids.append(line)
    return ids


def create_split(all_ids, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, 
                 random_seed=42, save_prefix=None):
    """Example of creating reproducible splits - for production use sklearn.
    
    This demonstrates the pattern, but in practice you should use:
    ```python
    from sklearn.model_selection import train_test_split
    
    # For train/test split
    train_ids, test_ids = train_test_split(
        all_ids, test_size=0.2, random_state=42
    )
    
    # For train/val/test split
    train_ids, temp_ids = train_test_split(
        all_ids, test_size=0.3, random_state=42
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.33, random_state=42  # 0.33 of 0.3 = 0.1
    )
    ```
    
    Args:
        all_ids: List of all sample IDs
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation  
        test_ratio: Fraction of data for testing
        random_seed: Seed for reproducible shuffling
        save_prefix: If provided, save splits to {prefix}_{split}.txt
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing ID lists
    """
    # Verify ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    assert abs(total_ratio - 1.0) < 1e-6, f"Ratios must sum to 1, got {total_ratio}"
    
    # Use sklearn for robust splitting
    temp_size = val_ratio + test_ratio
    train_ids, temp_ids = train_test_split(
        all_ids, test_size=temp_size, random_state=random_seed
    )
    
    # Split temp into val and test
    val_size_adjusted = val_ratio / temp_size
    val_ids, test_ids = train_test_split(
        temp_ids, train_size=val_size_adjusted, random_state=random_seed
    )
    
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    # Save if requested (this pattern is good to keep)
    if save_prefix:
        for split_name, split_ids in splits.items():
            filename = f"{save_prefix}_{split_name}.txt"
            with open(filename, 'w') as f:
                for id_ in split_ids:
                    f.write(f"{id_}\n")
            print(f"Saved {split_name} split to {filename} ({len(split_ids)} samples)")
    
    return splits


def verify_split(data_ids, split_ids, data_name="Your data"):
    """Verify that data matches a split definition.
    
    This helps catch common errors like:
    - Data in different order than split file
    - Missing samples
    - Duplicate samples
    - Extra samples not in split
    
    Args:
        data_ids: List of IDs from your data (in order)
        split_ids: List of IDs from split file (expected order)
        data_name: Name for display in messages
        
    Returns:
        True if data matches split exactly, False otherwise
        
    Example:
        >>> # Verify your features match the test split
        >>> feature_ids = [name for name, _ in loaded_features]
        >>> test_ids = load_split('test_ids.txt')
        >>> is_valid = verify_split(feature_ids, test_ids, "Feature file")
    """
    data_ids = list(data_ids)
    split_ids = list(split_ids)
    
    # Check length
    if len(data_ids) != len(split_ids):
        print(f"✗ Length mismatch: {data_name} has {len(data_ids)} samples, "
              f"split has {len(split_ids)} samples")
        return False
    
    # Check order
    order_mismatches = []
    for i, (data_id, split_id) in enumerate(zip(data_ids, split_ids)):
        if str(data_id) != str(split_id):
            order_mismatches.append((i, data_id, split_id))
    
    if order_mismatches:
        print(f"✗ Order mismatch: {len(order_mismatches)} samples in wrong order")
        # Show first few examples
        for i, data_id, split_id in order_mismatches[:3]:
            print(f"  Position {i}: got '{data_id}', expected '{split_id}'")
        if len(order_mismatches) > 3:
            print(f"  ... and {len(order_mismatches) - 3} more")
        return False
    
    # Check for duplicates in data
    data_set = set(data_ids)
    if len(data_set) != len(data_ids):
        print(f"✗ Duplicates found: {len(data_ids) - len(data_set)} duplicate IDs in {data_name}")
        return False
    
    print(f"✓ {data_name} matches split perfectly ({len(data_ids)} samples)")
    return True


# Example usage demonstration
def demo():
    """Demonstrate split management best practices."""
    print("=== Dataset Split Best Practices Demo ===\n")
    
    print("IMPORTANT: This demo shows the concepts, but for production use:")
    print("- Use sklearn.model_selection.train_test_split")
    print("- Use PyTorch's random_split for DataLoaders") 
    print("- Use established dataset classes (torchvision.datasets, etc.)\n")
    
    # Example 1: The pattern of saving/loading splits
    print("1. Key Concept - Reproducible Splits:")
    print("-" * 40)
    print("The most important practice is saving your split IDs!")
    print("This ensures all models use the exact same test set.\n")
    
    # Simulate a dataset with 1000 images
    fake_image_ids = [f"img_{i:04d}.jpg" for i in range(1000)]
    
    # Create splits using our example function
    splits = create_split(fake_image_ids, save_prefix="demo_dataset")
    
    print(f"\nSplit sizes:")
    for name, ids in splits.items():
        print(f"  {name}: {len(ids)} samples")
    
    # Example 2: Why verification matters
    print("\n2. Key Concept - Split Verification:")
    print("-" * 40)
    print("Always verify your data matches the expected split order!")
    
    # Load the test split we just created
    test_ids = load_split("demo_dataset_test.txt")
    
    # Show what happens with correct vs incorrect ordering
    print("\n✓ Correct ordering:")
    verify_split(test_ids, test_ids, "Features")
    
    print("\n✗ Common mistake - different ordering:")
    shuffled = test_ids.copy()
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
    verify_split(shuffled, test_ids, "Features")
    
    # Clean up demo files
    for split in ['train', 'val', 'test']:
        os.remove(f"demo_dataset_{split}.txt")
    
    print("\n=== Key Takeaways ===")
    print("1. Always save split IDs to files for reproducibility")
    print("2. Use established tools (sklearn, PyTorch) for splitting")
    print("3. Verify data order matches split files before evaluation")
    print("4. Version control your split files!")


if __name__ == '__main__':
    # Run the demo
    demo()
    
    print("\n" + "="*60)
    print("For Production Use:")
    print("="*60)
    print("""
# Better approach using established tools:
from sklearn.model_selection import train_test_split
import pandas as pd

# Example with pandas and sklearn:
df = pd.read_csv('dataset.csv')
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv('train_split.csv', index=False)
test_df.to_csv('test_split.csv', index=False)

# Example with PyTorch:
from torch.utils.data import random_split
dataset = YourDataset()
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# The key is: whatever method you use, save the splits for reproducibility!
""")