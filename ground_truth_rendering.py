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
        self.distance = 2.73
        self.deg_per_view = 360 / views
        self.max_dimension = 1.0
        self.renderer = sr.SoftRenderer(image_size=image_size,
                            sigma_val=sigma_val, aggr_func_rgb='hard',
                            camera_mode='look_at',
                            viewing_angle=360 / views,
                            dist_eps=1e-10)

    def render_ground_truth(self, path, chunk=6):
        # get mesh
        # import ipdb;ipdb.set_trace()
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
            images.append(self.renderer(v, f).detach())
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


if __name__ == '__main__':
    from os import listdir, mkdir
    from os.path import isdir, isfile
    import cv2
    views = 12
    gtr = GroundTruthRenderer(64, 1e-4, views)
    rootfolder = 'data/MN40Objs/'
    outdir = 'data/MN40Views/'
    outtxtfile = 'data/MN40_train.txt'
    outtxtfile2 = 'data/MN40_test.txt'
    models = listdir(rootfolder)
    models = [ff for ff in models if isdir(rootfolder+ff)]
    models.sort()
    cats = ['train', 'test']
    f1 = open(outtxtfile, 'w')
    f2 = open(outtxtfile2, 'w')
    for mm in models:
        outmodelfolder = outdir + mm
        print(outmodelfolder)
        if not isdir(outmodelfolder):
            mkdir(outmodelfolder)
        for cc in cats:
            datafolder = rootfolder + mm + '/' + cc 
            outdatafolder = outmodelfolder+ '/' + cc
            print(outdatafolder)
            if not isdir(outdatafolder):
                mkdir(outdatafolder)
            files = listdir(datafolder)
            files = [ff for ff in files if ff.endswith('.obj')]
            files.sort()

            for ff in files:
                filename = datafolder+ '/' + ff
                outfilename = outdatafolder  + '/' + ff.split('.obj')[0]
                if cc == 'train':
                    f1.write(outfilename+'\n')
                else:
                    f2.write(outfilename+'\n')
                images, viewpoints = gtr.render_ground_truth(filename)
                imagenp = images.detach().cpu().numpy().transpose((0, 2, 3, 1))
                imagenp = (255*imagenp).astype(np.uint8)

                for k in range(views):
                    cv2.imwrite(outfilename+'_'+str(k).zfill(3)+'.png', imagenp[k])

    f1.close()
    f2.close()
    # import ipdb;ipdb.set_trace()
