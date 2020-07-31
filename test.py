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
import os
from ground_truth_rendering import GroundTruthRenderer


resume_path = "data/results/models/2020-07-30-09:02:57.021496/" + \
              "checkpoint_0005300.pth.tar"

SIGMA_VAL = 1e-4
IMAGE_SIZE = 64
VIEWS = 12

parser = argparse.ArgumentParser()
parser.add_argument('-is', '--image-size', type=int, default=IMAGE_SIZE)
parser.add_argument('-sv', '--sigma-val', type=float, default=SIGMA_VAL)
args = parser.parse_args()


start_iter = int(os.path.split(resume_path)[1][11:].split('.')[0]) + 1
print('Loading model from iteration %s' % start_iter)
model = models.Model('data/obj/sphere/sphere_1352.obj', args=args)
model = nn.DataParallel(model)
model = model.cuda()
state_dicts = torch.load(resume_path)
model.load_state_dict(state_dicts['model'])


dataset_directory = 'data/MN40Objs'

gtr = GroundTruthRenderer(IMAGE_SIZE, SIGMA_VAL, VIEWS)
dataset_test = datasets.ModelNet40(dataset_directory, partition='test',
                                   truncate = None)
test_loader = DataLoader(dataset_test, batch_size = 16, shuffle = False)
dataset_train = datasets.ModelNet40(dataset_directory, partition='train',
                                    truncate = None)
test_loader = DataLoader(dataset_test, batch_size = 16, shuffle = True)

cat_map = dataset_train.categories

def get_svm_data(data_loader):
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

        feats = model.module.compute_features(images).detach().cpu()
        feats = [i for i in feats]
        assert all(len(feat) == model.module.encoder.dim_out for feat in feats)

        answers = [cat_map[j] for j in categories]
        x_data += feats
        y_data += answers

    return x_data, y_data


def test_accuracy():
    # svm_train_feats, svm_train_labels = get_svm_data(train_loader)
    svm_test_feats, svm_test_labels = get_svm_data(test_loader)
    classifier = svm.LinearSVC()
    print("Fitting SVM")
    classifier.fit(svm_test_feats, svm_test_labels)

    predicted_labels = classifier.predict(svm_test_feats)
    acc = sum(svm_test_labels == predicted_labels) / len(predicted_labels)
    return acc

print(test_accuracy())
