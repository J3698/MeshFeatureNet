import soft_renderer as sr
import soft_renderer.functional as srf
import numpy as np
import torch

class GroundTruthRenderer():
    def __init__(self, image_size, sigma_val, views):
        self.image_size = image_size
        self.sigma_val = sigma_val
        self.views = views
        self.elevation = 30.
        self.distance = 2.732
        self.deg_per_view = 360 / views
        self.max_dimension = 0.5
        self.renderer = sr.SoftRenderer(image_size=image_size,
                            sigma_val=sigma_val, aggr_func_rgb='hard',
                            camera_mode='look_at',
                            viewing_angle=360 / views,
                            dist_eps=1e-10)

    def render_ground_truth(self, path, chunk=6):
        # get mesh
        mesh = sr.Mesh.from_obj(path)
        self.center_vertices(mesh.vertices)
        self.scale_vertices(mesh.vertices, self.max_dimension)
        vertices = torch.chunk(torch.cat(self.views * [mesh.vertices]), chunk)
        faces = torch.chunk(torch.cat(self.views * [mesh.faces]), chunk)

        # get viewpoints
        distances = torch.ones(self.views).float() * self.distance
        elevations = torch.ones(self.views).float() * self.elevation
        rotations = (-torch.arange(0, self.views) * self.deg_per_view).float()
        viewpoints = srf.get_points_from_angles(distances, elevations, rotations)
        viewpoints = torch.chunk(viewpoints, chunk)
        images = []
        # render images
        for views, v, f in zip(viewpoints, vertices, faces):
            self.renderer.transform.set_eyes(views)
            images.append(self.renderer(v, f))
        return torch.cat(images), torch.cat(viewpoints)


    def center_vertices(self, vertices):
        get_center = lambda ax: (vertices[0, :, ax].max().item() + 
                                 vertices[0, :, ax].min().item()) / 2
        for ax in range(vertices.shape[2]):
            vertices[:, :, ax] -= get_center(ax)


    def scale_vertices(self, vertices, max_dimension):
        max_dim = vertices.abs().max().item()
        scale_factor = max_dimension / max_dim
        vertices *= scale_factor


