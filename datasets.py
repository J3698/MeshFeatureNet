import os
import soft_renderer as sr
from random import shuffle
import torch
from torch.utils.data import Dataset
import numpy as np
import tqdm
from utils import imgs_to_gif

class ModelNet40(Dataset):
    def __init__(self, folder, partition = None, truncate = None, reverse = False,
                 random = False):
        print("Constructing dataset")
        self.partition = partition
        self.folder = folder
        self.paths = sorted(self.load_file_paths())
        if random:
            shuffle(self.paths)
        if reverse:
            print("Reversed")
            self.paths.reverse()
        if not truncate is None:
            self.paths = self.paths[:min(len(self.paths), truncate)]
        self.categories = self.get_categories()
        print(self)


    def __getitem__(self, idx):
        return self.paths[idx]


    def __len__(self):
        return len(self.paths)


    def __repr__(self):
        template = ("Partition: {}\nFolder: {}\nModels: {}"
                    "\nSample model path: {}\n")
        sample_path = "N/A" if len(self.paths) == 0 else self.paths[0]
        return template.format(self.partition, self.folder,
                               len(self), sample_path)


    def load_file_paths(self):
        data = []
        for root, dirs, files in os.walk(self.folder):
            data += [os.path.join(root, f) for f in files]
        return list(filter(self.filter_partition, data))


    def filter_partition(self, filepath):
        if self.partition is None:
            return True

        folder = os.path.split(filepath)[0]
        partition = os.path.split(folder)[1]
        return partition == self.partition


    def get_categories(self):
        categories = sorted(list(set(self.get_category(path) for path in self.paths)))
        return dict((cat, idx) for idx, cat in enumerate(categories))


    def get_category(self, path):
        partition_folder = os.path.split(path)[0]
        category_folder = os.path.split(partition_folder)[0]
        category = os.path.split(category_folder)[1]
        return category
