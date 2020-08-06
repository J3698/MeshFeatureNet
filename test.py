import argparse

import torch
from sklearn import svm
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from losses import multiview_iou_loss
from utils import AverageMeter, imgs_to_gif
import soft_renderer as sr
import soft_renderer.functional as srf
from tqdm import tqdm
import datasets
import models
import imageio
import time
from datetime import datetime
from time import sleep
import os
import sys
from ground_truth_rendering import GroundTruthRenderer


model1_path = "/data/datasets/asanders/MeshFeatureNet/data/results/models/2020-08-03-13:44:28.549495"
model2_path = "/data/datasets/asanders/MeshFeatureNet/data/results/models/2020-08-03-13:42:10.827513"

dataset_directory = 'data/MN40Objs'

SIGMA_VAL = 1e-4
IMAGE_SIZE = 64
VIEWS = 12

parser = argparse.ArgumentParser()
parser.add_argument('-is', '--image-size', type=int, default=IMAGE_SIZE)
parser.add_argument('-sv', '--sigma-val', type=float, default=SIGMA_VAL)
args = parser.parse_args()

gtr = GroundTruthRenderer(IMAGE_SIZE, SIGMA_VAL, VIEWS)

dataset_test = datasets.ModelNet40(dataset_directory, partition='test',
                                   truncate = None)
test_loader = DataLoader(dataset_test, batch_size = 16, shuffle = False)

dataset_train = datasets.ModelNet40(dataset_directory, partition='train',
                                    truncate = None)
train_loader = DataLoader(dataset_train, batch_size = 16, shuffle = False)

cat_map = dataset_train.categories



def main():
    model1 = models.Model('data/obj/sphere/sphere_1352.obj', args=args)
    model1 = nn.DataParallel(model1)
    model1 = model1.cuda()

    model2 = models.Model('data/obj/sphere/sphere_642.obj', args=args)
    model2 = nn.DataParallel(model2)
    model2 = model2.cuda()

    to_compare = zip(get_checkpoints(model1, model1_path), get_checkpoints(model2, model2_path))
    for modela, modelb in to_compare:
        print("1", test_accuracy(modela))
        print("2", test_accuracy(modelb))


def get_checkpoints(model, model_path):
    paths = sorted(os.listdir(model_path))
    for checkpoint in paths:
        if checkpoint.endswith("pth.tar"):
            resume_path = os.path.join(model_path, checkpoint)
            print(resume_path)
            state_dicts = torch.load(resume_path)
            model.load_state_dict(state_dicts['model'])
            yield model


def get_svm_data(model, data_loader):
    print("Getting SVM features")
    x_data = []
    y_data = []
    for i, paths in tqdm(enumerate(data_loader)):
        data = [gtr.render_ground_truth(path, chunk=12) for path in paths]
        images, viewpoints = zip(*data)
        images = torch.cat([k.unsqueeze(0) for k in images], axis = 0)
        viewpoints = torch.cat([v.unsqueeze(0) for v in viewpoints], axis = 0)
        categories = [dataset_train.get_category(i) for i in paths]
        batch_size = len(images)
        assert batch_size == len(categories) and batch_size == len(viewpoints)

        feats = model.module.compute_features(images).detach().cpu().numpy()
        feats = [i.tolist() for i in feats]
        assert all(len(feat) == model.module.encoder.dim_out for feat in feats)

        answers = [cat_map[j] for j in categories]
        x_data += feats
        y_data += answers

    return x_data, y_data


def test_accuracy(model):
    svm_train_feats, svm_train_labels = get_svm_data(model, train_loader)
    svm_test_feats, svm_test_labels = get_svm_data(model, test_loader)
    classifier = svm.LinearSVC()
    print("Fitting SVM")
    classifier.fit(svm_train_feats, svm_train_labels)

    predicted_labels = classifier.predict(svm_test_feats)
    acc = sum(svm_test_labels == predicted_labels) / len(predicted_labels)
    return acc

if __name__ == "__main__":
    main()
