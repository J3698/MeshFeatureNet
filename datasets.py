import os
import soft_renderer.functional as srf
import soft_renderer as sr
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tqdm
from utils import imgs_to_gif

class ModelNet40(Dataset):
    def __init__(self, folder, image_size, sigma_val, num_views, partition=None):
        """
        Dataset returns rendered images of object
        """
        print("Constructing dataset")
        self.partition = partition
        self.folder = folder
        self.paths = self.load_file_paths()
        self.elevation = 30.
        self.distance = 2.732
        self.max_model_dimension = .5
        self.num_views = num_views
        self.deg_per_view = 360 / self.num_views
        self.image_size = image_size
        self.sigma_val = sigma_val
    

        self.categories = self.get_categories()
        print(self)
        # self.save_test_render()
        self.categories = dict((cat, idx) for idx, cat in 
                            enumerate(self.categories))


    def __repr__(self):
        template = ("Partition: {}\nFolder: {}\nModels: {}"
                    "\nSample model path: {}\n")
        return template.format(self.partition, self.folder,
                               len(self), self.paths[0])


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


    def save_test_render(self, idx = 0):
        imgs = torch.chunk(self[idx][0], self.num_views, dim = 0)
        imgs_to_gif(imgs, 'rotation{}.gif'.format(idx))


    def get_categories(self):
        return set(self.get_category(i) for i in range(len(self)))


    def get_category(self, idx):
        path = self.paths[idx]
        partition_folder = os.path.split(path)[0]
        category_folder = os.path.split(partition_folder)[0]
        category = os.path.split(category_folder)[1]
        return category


    def __getitem__(self, idx):
        return  self.paths[idx]



    def get_meshes_for_views(self, idx):

        self.center_vertices(vertices)
        self.scale_vertices(vertices)
        return faces, vertices


    def center_vertices(self, vertices):
        get_center = lambda ax: (vertices[0, :, ax].max().item() + 
                                 vertices[0, :, ax].min().item()) / 2
        for ax in range(vertices.shape[2]):
            vertices[:, :, ax] -= get_center(ax)


    def scale_vertices(self, vertices):
        max_dim = vertices.abs().max().item()
        scale_factor = self.max_model_dimension / max_dim
        vertices *= scale_factor


    def get_surrounding_viewpoints(self):
        return viewpoints


    def render_images(self, faces, vertices, viewpoints):
        return images


    def __len__(self):
        return len(self.paths)
