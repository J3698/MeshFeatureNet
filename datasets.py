import os
import soft_renderer.functional as srf
import soft_renderer as sr
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tqdm
import imageio

class ModelNet40():
    def __init__(self, folder, image_size, sigma_val, partition=None):
        """
        Dataset returns rendered images of object
        """
        print("Constructing dataset")
        self.partition = partition
        self.folder = folder
        self.paths = self.load_file_paths()
        self.elevation = 30.
        self.distance = 2.732
        self.num_views = 24
        self.deg_per_view = 360 / self.num_views
        self.renderer = sr.SoftRenderer(image_size=image_size,
                                sigma_val=sigma_val, aggr_func_rgb='hard',
                                camera_mode='look_at',
                                viewing_angle=self.deg_per_view,
                                dist_eps=1e-10)
        self.save_test_render()
        print()
        print(self.partition)
        print(self.folder)
        print(len(self))
        print(self.paths[0])


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

    def save_test_render(self):
        writer = imageio.get_writer('rotation.gif', mode='I')
        images = torch.chunk(self[0][0], 24, dim = 0)
        for idx, img in enumerate(images):
            img = img.cpu().numpy().squeeze()
            print(img.shape)
            writer.append_data((255 * img.transpose((1, 2, 0))).astype(np.uint8))
        writer.close()

    def get_random_batch(self, batch_size):
        images_and_viewpoints = \
            [self[np.random.randint(0, len(self) - 1)] for i in range(batch_size)]
        images, viewpoints = zip(*images_and_viewpoints)
        return torch.cat(images, dim = 0), torch.cat(viewpoints, dim = 0)


    def __getitem__(self, idx):
        faces, vertices = self.get_meshes_for_views(idx)
        print("faces, vertices", faces.shape, vertices.shape)

        viewpoints = self.get_surrounding_viewpoints()
        print("viewpoints", viewpoints.shape)
        images = self.render_images(faces, vertices, viewpoints)
        print("images", images.shape)

        return images, viewpoints


    def get_meshes_for_views(self, idx):
        mesh = sr.Mesh.from_obj(self.paths[idx])
        faces = torch.cat(self.num_views * [mesh.faces.cuda()])
        vertices = torch.cat(self.num_views * [mesh.vertices.cuda()])
        return faces, vertices


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
