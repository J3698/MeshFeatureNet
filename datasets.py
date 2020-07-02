import os
import soft_renderer.functional as srf
import soft_renderer as sr
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tqdm
from utils import imgs_to_gif

class ModelNet40():
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
        self.renderer = sr.SoftRenderer(image_size=image_size,
                                sigma_val=sigma_val, aggr_func_rgb='hard',
                                camera_mode='look_at',
                                viewing_angle=self.deg_per_view,
                                dist_eps=1e-10)
        print(self)
        self.save_test_render()
    
        self.categories = self.get_categories()
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
        imgs = torch.chunk(self[idx][0], 24, dim = 0)
        imgs_to_gif(imgs, 'rotation{}.gif'.format(idx))


    """
    def get_random_batch(self, batch_size):
        random_obj_id = lambda: np.random.randint(0, len(self) - 1)
        datapoints = [self[random_obj_id()] for i in range(batch_size)]
        images, viewpoints, categories = zip(*datapoints)

        images = torch.cat(images, dim = 0)
        images = images.unsqueeze(0)
        imges = images.reshape(batch_size, self.num_views, 4, 
                               self.image_size, self.image_size)
        viewpoints = torch.cat(viewpoints, dim = 0)
        viewpoints = viewpoints.unsqueeze(0)
        viewpoints = viewpoints.reshape(batch_size, self.num_views, 3)

        return imges, viewpoints, categories
    """


    def __getitem__(self, idx):
        faces, vertices = self.get_meshes_for_views(idx)


        viewpoints = self.get_surrounding_viewpoints()
        images = self.render_images(faces, vertices, viewpoints)
        category = self.get_category(idx)

        del faces
        del vertices

        return images, viewpoints, category


    def get_categories(self):
        return set(self.get_category(i) for i in range(len(self)))


    def get_category(self, idx):
        path = self.paths[idx]
        partition_folder = os.path.split(path)[0]
        category_folder = os.path.split(partition_folder)[0]
        category = os.path.split(category_folder)[1]
        return category

    def get_meshes_for_views(self, idx):
        try:
            mesh = sr.Mesh.from_obj(self.paths[idx])
        except UnicodeDecodeError as e:
            print(self.paths[idx])
            raise e from None

        vertices = torch.cat(self.num_views * [mesh.vertices.cuda()])
        faces = torch.cat(self.num_views * [mesh.faces.cuda()])

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
        distances = torch.ones(self.num_views).float() * self.distance
        elevations = torch.ones(self.num_views).float() * self.elevation
        rotations = (-torch.arange(0, self.num_views) * self.deg_per_view).float()
        viewpoints = srf.get_points_from_angles(distances, elevations, rotations)
        return viewpoints.cuda()


    def render_images(self, faces, vertices, viewpoints):
        self.renderer.transform.set_eyes(viewpoints)
        images = self.renderer(vertices, faces)
        return images


    def __len__(self):
        return len(self.paths)
