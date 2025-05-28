"""
Flickr30K Dataset class for Query2Label integration.
This file should be placed in the query2label/dataset/ directory.
"""

from torch.utils.data import Dataset
from PIL import Image
import os


class Flickr30kDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])

        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error occurred when loading image {img_name}: {e}")
            return None

        if self.transform:
            image = self.transform(image)

        return image, self.image_list[idx]