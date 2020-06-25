import os
import soft_renderer.functional as srf
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tqdm

class ModelNet40():
    def __init__(self, folder, image_size, sigma_val, partition=None):
        """
        Dataset returns rendered images of object
        """
        self.partition = partition
        self.paths = self.load_file_paths()
        self.renderer = sr.SoftRenderer(image_size=image_size,
                                sigma_val=sigma_val, aggr_func_rgb='hard',
                                camera_mode='look_at', viewing_angle=15,
                                dist_eps=1e-10)

    def __getitem__(self, idx):
        mesh = sr.Mesh.from_obj(filename_obj)
        vertices = self.template_mesh.vertices.cpu()[0]
        faces = self.template_mesh.faces.cpu()[0]
        silhouettes = self.renderer(vertices, faces)
        self.renderer.transform.set_eyes(viewpoints)

    def load_file_paths(self):
        self.data = []
        for root, dirs, files in os.walk(folder):
            self.data += [os.path.join(root, f) for f in files]
        self.data = list(filter(self.filter_partition, self.data))

    def filter_partition(self, filepath):
        if self.partition is None:
            return True

        folder = os.path.split(filepath)[0]
        partition_folder = os.path.split(folder)[0]
        partition = os.path.split(partition_folder)[1]
        return partition == self.partition
