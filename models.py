import torch
import torch.nn as nn
import torch.nn.functional as F

import soft_renderer as sr
import soft_renderer.functional as srf
import math

class Encoder(nn.Module):
    def __init__(self, dim_in=4, dim_out=512, dim1=64, dim2=1024, im_size=64):
        super(Encoder, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim1 = dim1
        self.dim2 = dim2
        self.im_size = im_size
        dim_hidden = [dim1, dim1*2, dim1*4, dim2, dim2]

        self.conv1 = nn.Conv2d(dim_in, dim_hidden[0], kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(dim_hidden[0], dim_hidden[1], kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(dim_hidden[1], dim_hidden[2], kernel_size=5, stride=2, padding=2)

        self.bn1 = nn.BatchNorm2d(dim_hidden[0])
        self.bn2 = nn.BatchNorm2d(dim_hidden[1])
        self.bn3 = nn.BatchNorm2d(dim_hidden[2])

        self.fc1 = nn.Linear(dim_hidden[2]*math.ceil(im_size/8)**2, dim_hidden[3])
        self.fc2 = nn.Linear(dim_hidden[3], dim_hidden[4])
        self.fc3 = nn.Linear(dim_hidden[4], dim_out)

        self.encoder = nn.LSTM(dim_out, dim_out, num_layers = 2)

    def forward(self, x):
        # convert batch of image sequences into one big batch
        batch_size, num_views, image_channels, image_size, _ = x.shape
        x = x.reshape(batch_size * num_views, image_channels, image_size, image_size)

        # extract features from images
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.relu(self.fc3(x), inplace=True)

        # convert big batch back into batch of sequences
        x = x.reshape(batch_size, num_views, -1)

        # encode views with LSTM
        x = x.transpose(0, 1)
        self.encoder.flatten_parameters()
        x = self.encoder(x)[0][-1]
        assert x.shape == (batch_size, self.dim_out)
        # x = x[:,0,:]

        return x


class Decoder(nn.Module):
    def __init__(self, filename_obj, dim_in=512, centroid_scale=0.1, bias_scale=1.0, centroid_lr=0.1, bias_lr=1.0):
        super(Decoder, self).__init__()
        # load .obj
        self.template_mesh = sr.Mesh.from_obj(filename_obj)
        # vertices_base, faces = srf.load_obj(filename_obj)
        self.register_buffer('vertices_base', self.template_mesh.vertices.cpu()[0])#vertices_base)
        self.register_buffer('faces', self.template_mesh.faces.cpu()[0])#faces)

        self.nv = self.vertices_base.size(0)
        self.nf = self.faces.size(0)
        self.centroid_scale = centroid_scale
        self.bias_scale = bias_scale
        self.obj_scale = .5

        dim = 1024
        dim_hidden = [dim, dim*2]
        self.fc1 = nn.Linear(dim_in, dim_hidden[0])
        self.fc2 = nn.Linear(dim_hidden[0], dim_hidden[1])
        self.fc_centroid = nn.Linear(dim_hidden[1], 3)
        self.fc_bias = nn.Linear(dim_hidden[1], self.nv*3)


    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        # decoder follows NMR
        centroid = self.fc_centroid(x) * self.centroid_scale

        bias = self.fc_bias(x) * self.bias_scale
        bias = bias.view(-1, self.nv, 3)

        base = self.vertices_base * self.obj_scale

        sign = torch.sign(base)
        base = torch.abs(base)
        base = torch.log(base / (1 - base))

        centroid = torch.tanh(centroid[:, None, :])
        scale_pos = 1 - centroid
        scale_neg = centroid + 1

        vertices = torch.sigmoid(base + bias) * sign
        vertices = F.relu(vertices) * scale_pos - F.relu(-vertices) * scale_neg
        vertices = vertices + centroid
        vertices = vertices * 0.5
        faces = self.faces[None, :, :].repeat(batch_size, 1, 1)

        return vertices, faces


class Model(nn.Module):
    def __init__(self, filename_obj, args):
        super(Model, self).__init__()

        self.args = args
        self.encoder = Encoder(im_size=args.image_size)
        self.decoder = Decoder(filename_obj)
        self.renderer = sr.SoftRenderer(image_size=args.image_size, sigma_val=args.sigma_val, 
                                        aggr_func_rgb='hard', camera_mode='look_at',
                                        viewing_angle=15, dist_eps=1e-10)
        self.laplacian_loss = sr.LaplacianLoss(self.decoder.vertices_base, self.decoder.faces)
        self.flatten_loss = sr.FlattenLoss(self.decoder.faces)


    def model_param(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())


    def set_sigma(self, sigma):
        self.renderer.set_sigma(sigma)


    def compute_features(self, images):
        # assert shape
        batch_size = images.shape[0]
        num_views = images.shape[1]
        img_size = self.args.image_size
        # 4 for rgba
        assert images.shape == (batch_size, num_views, 4, img_size, img_size) 

        # compute features
        feats = self.encoder(images)
        assert feats.shape == (batch_size, self.encoder.dim_out)
        return feats


    def reconstruct(self, images):
        """
        Takes a batch of image sequences and a batch of viewpoint sequences, and attempts to
        predict the meshes

        Returns the predicted mesh vertices and faces
        """
        vertices, faces = self.decoder(self.encoder(images))
        return vertices, faces


    def predict_multiview(self, images, viewpoints):
        """
        Takes a batch of image sequences and a batch of viewpoint sequences, and attempts to
        reconstruct the images by predicting and then rendering meshes.

        Returns the reconstructed images and the laplacian and flatten losses from the predicted
        meshes.
        """
        # import ipdb;ipdb.set_trace()
        batch_size, num_views, _, _, _ = images.shape

        # generate meshes
        vertices, faces = self.reconstruct(images)
        num_vertices, num_faces = vertices.shape[1], faces.shape[1]
        assert vertices.shape == (batch_size, num_vertices, 3)
        assert faces.shape == (batch_size, num_faces, 3)

        # calculate mesh loss
        # wenshan: calculate these two losses before replication
        laplacian_loss = self.laplacian_loss(vertices)
        flatten_loss = self.flatten_loss(vertices)

        # copy meshes for batch rendering
        vertices = torch.cat(num_views * [vertices.unsqueeze(0)])
        faces = torch.cat(num_views * [faces.unsqueeze(0)])
        assert vertices.shape == (num_views, batch_size, num_vertices, 3)
        assert faces.shape == (num_views, batch_size, num_faces, 3)

        # transpose and reshape meshes for batch rendering
        vertices = vertices.transpose_(0, 1).reshape(num_views * batch_size, -1, 3)
        faces = faces.transpose_(0, 1).reshape(num_views * batch_size, -1, 3)
        assert vertices.shape == (batch_size * num_views, num_vertices, 3)
        assert faces.shape == (batch_size * num_views, num_faces, 3)

        # reshape viewpoints for batch rendering
        viewpoints = viewpoints.reshape(-1, 3)
        assert viewpoints.shape == (batch_size * num_views, 3)

        # render images from meshes
        self.renderer.transform.set_eyes(viewpoints)
        reconstructed_images = self.renderer(vertices, faces)


        # return images as separate tensors
        batch_size = images.size(0)
        return reconstructed_images, laplacian_loss, flatten_loss


    def forward(self, images, viewpoints):
        return self.predict_multiview(images, viewpoints)
