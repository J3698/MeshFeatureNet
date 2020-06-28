import os
import soft_renderer.functional as srf
import soft_renderer as sr
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
        self.folder = folder
        self.paths = self.load_file_paths()
        self.elevation = 30.
        self.distance = 2.732
        self.renderer = sr.SoftRenderer(image_size=image_size,
                                sigma_val=sigma_val, aggr_func_rgb='hard',
                                camera_mode='look_at', viewing_angle=15,
                                dist_eps=1e-10)

    def get_random_batch(self, batch_size):
        imgs, viewpoints = [], []
        for i in range(batch_size):
            new_imgs, new_viewpoints = self[np.random.randint(0, len(self) - 1)]
            imgs.append(new_imgs)
            viewpoints.append(new_viewpoints)
        return imgs, viewpoints

    def __getitem__(self, idx):
        mesh = sr.Mesh.from_obj(self.paths[idx])
        vertices = mesh.vertices.cpu()[0]
        print("v", vertices.shape)
        faces = mesh.faces.cpu()[0]
        print("f", faces.shape)
        distances = torch.ones(24).float() * self.distance
        elevations = torch.ones(24).float() * self.elevation
        rotations = (-torch.arange(0, 24) * 15).float()
        viewpoints = srf.get_points_from_angles(distances, elevations, rotations)
        print("viewpoints", viewpoints.shape)
        self.renderer.transform.set_eyes(viewpoints)
        vertices = torch.cat([vertices for i in range(12)], dim = 0)
        faces = torch.cat([faces for i in range(12)], dim = 0)
        silhouettes = self.renderer(vertices, faces)
        imgs = silhouettes.chunk(24, dim=0)
        return imgs, viewpoints

    def __len__(self):
        return len(self.paths)

    def load_file_paths(self):
        data = []
        for root, dirs, files in os.walk(self.folder):
            data += [os.path.join(root, f) for f in files]
        return list(filter(self.filter_partition, data))

    def filter_partition(self, filepath):
        if self.partition is None:
            return True

        folder = os.path.split(filepath)[0]
        partition_folder = os.path.split(folder)[0]
        partition = os.path.split(partition_folder)[1]
        return partition == self.partition
