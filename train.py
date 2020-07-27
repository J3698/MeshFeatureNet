print("Importing")

import argparse

import torch
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

print("Parsing args")

CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
LR_TYPE = 'step'

LAMBDA_LAPLACIAN = 5e-3
LAMBDA_FLATTEN = 5e-4

PRINT_FREQ = 5
DEMO_FREQ = 50
SAVE_FREQ = 100
RANDOM_SEED = 0

MODEL_DIRECTORY = 'data/results/models'
DATASET_DIRECTORY = 'data/MN40Objs'

IMAGE_SIZE = 64
SIGMA_VAL = 1e-4
START_ITERATION = 0

EPOCHS = 10000

VIEWS = 12

RESUME_PATH = ''

TIME = str(datetime.now()).replace(" ", "-")

TRAIN_TRUNCATION = 16

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('-eid', '--experiment-id', type=str, default=TIME)
parser.add_argument('-tt', '--train-truncation', type=str, default=TRAIN_TRUNCATION)
parser.add_argument('-md', '--model-directory', type=str, default=MODEL_DIRECTORY)
parser.add_argument('-r', '--resume-path', type=str, default=RESUME_PATH)
parser.add_argument('-dd', '--dataset-directory', type=str, default=DATASET_DIRECTORY)
parser.add_argument('-cls', '--class-ids', type=str, default=CLASS_IDS_ALL)
parser.add_argument('-is', '--image-size', type=int, default=IMAGE_SIZE)
parser.add_argument('-v', '--views', type=int, default=VIEWS)
parser.add_argument('-e', '--epochs', type=int, default=EPOCHS)
parser.add_argument('-b', '--batch-size', type=int, default=BATCH_SIZE)

parser.add_argument('-sv', '--sigma-val', type=float, default=SIGMA_VAL)

parser.add_argument('-lr', '--learning-rate', type=float, default=LEARNING_RATE)
parser.add_argument('-lrt', '--lr-type', type=str, default=LR_TYPE)

parser.add_argument('-ll', '--lambda-laplacian', type=float, default=LAMBDA_LAPLACIAN)
parser.add_argument('-lf', '--lambda-flatten', type=float, default=LAMBDA_FLATTEN)
parser.add_argument('-pf', '--print-freq', type=int, default=PRINT_FREQ)
parser.add_argument('-df', '--demo-freq', type=int, default=DEMO_FREQ)
parser.add_argument('-sf', '--save-freq', type=int, default=SAVE_FREQ)
parser.add_argument('-s', '--seed', type=int, default=RANDOM_SEED)

args = parser.parse_args()
print(args)

print("Setting up torch/np")

torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

directory_output = os.path.join(args.model_directory, args.experiment_id)
os.makedirs(directory_output, exist_ok=True)
image_output = os.path.join(directory_output, 'pic')
os.makedirs(image_output, exist_ok=True)

# setup model & optimizer
model = models.Model('data/obj/sphere/sphere_1352.obj', args=args)

print("GPUs: {}".format(torch.cuda.device_count()))
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.cuda()

optimizer = torch.optim.Adam(model.module.model_param(), args.learning_rate)

start_iter = START_ITERATION
if args.resume_path:
    print("Loading resume information")
    state_dicts = torch.load(args.resume_path)
    model.load_state_dict(state_dicts['model'])
    optimizer.load_state_dict(state_dicts['optimizer'])
    start_iter = int(os.path.split(args.resume_path)[1][11:].split('.')[0]) + 1
    print('Resuming from %s iteration' % start_iter)
 
print()

dataset_train = datasets.ModelNet40(args.dataset_directory, partition='train', truncate = args.train_truncation, reverse = True)
dataset_test = datasets.ModelNet40(args.dataset_directory, partition='test')


test_loader = DataLoader(dataset_test, batch_size = args.batch_size, shuffle = False)
train_loader = DataLoader(dataset_train, batch_size = args.batch_size, shuffle = True)


gtr = GroundTruthRenderer(args.image_size, args.sigma_val, args.views)

cat_map = dataset_train.categories


def get_svm_data(data_loader):
    print("Getting SVM features")
    x_data = []
    y_data = []
    for i, data in tqdm(enumerate(data_loader)):
        images, viewpoints, categories = data
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
        

def train():
    print("Starting to train")

    end = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    batch_num = 0
    for e in range(args.epochs):
        for j, paths in enumerate(train_loader):
            data = [gtr.render_ground_truth(path) for path in paths]
            images, viewpoints = zip(*data)
            images = torch.cat([k.unsqueeze(0) for k in images], axis = 0)
            viewpoints = torch.cat([v.unsqueeze(0) for v in viewpoints], axis = 0)
            categories = None
            train_batch(images, viewpoints, categories, losses, batch_num, e, batch_time)
            batch_time.update(time.time() - end)

            batch_num += 1
            end = time.time()
            del images
            del viewpoints



def train_batch(images, viewpoints, categories, losses, i, e, batch_time):
    # forward
    lr = adjust_learning_rate([optimizer], args.learning_rate,
                              i, method=args.lr_type)
    model.module.set_sigma(adjust_sigma(args.sigma_val, i))
    model_images, laplacian_loss, flatten_loss = model(images, viewpoints)
    laplacian_loss_avg = laplacian_loss.mean()
    flatten_loss_avg = flatten_loss.mean()
    images_reshaped = images.reshape(model_images.shape)
    assert images_reshaped.shape == (args.batch_size * args.views, 4,
                                     args.image_size, args.image_size)

    # compute loss
    loss = multiview_iou_loss(images, model_images) + \
           args.lambda_laplacian * laplacian_loss_avg + \
           args.lambda_flatten * flatten_loss_avg
    losses.update(loss.data.item(), model_images.size(0))

    # compute gradient and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 100 == 0 and i != 0:
        pass
        # print(test_accuracy())
    if i % args.save_freq == 0:
        save_checkpoint(i)
    if i % args.demo_freq == 0:
        save_demo_images(images_reshaped, model_images, i)
    if i % args.print_freq == 0:
        print_iteration_info(i, e, batch_time, losses, lr)

    del loss
    del laplacian_loss
    del flatten_loss
    del laplacian_loss_avg
    del flatten_loss_avg
    del images
    del model_images
    del images_reshaped

def print_iteration_info(i, epoch, batch_time, losses, lr):
    print('Iter: [{0}, {1}]\t'
          'Time {batch_time.val:.3f}\t'
          'Loss {loss.val:.3f}\t'
          'lr {lr:.6f}\t'
          'sv {sv:.6f}\t'.format(i, epoch,
                                 batch_time=batch_time, loss=losses, 
                                 lr=lr, sv=model.module.renderer.rasterizer.sigma_val))


def save_demo_images(images, model_images, i):
    demo_input_images = images[0:args.views]
    demo_fake_images = model_images[0:args.views]
    fake_img_path = os.path.join(image_output,'%07d_fake.gif' % i)
    input_img_path = os.path.join(image_output, '%07d_input.gif' % i)
    imgs_to_gif(demo_fake_images, fake_img_path)
    imgs_to_gif(demo_input_images, input_img_path)


def save_checkpoint(i):
    model_path = os.path.join(directory_output, 'checkpoint_%07d.pth.tar'%i)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        }, model_path)


def adjust_learning_rate(optimizers, learning_rate, i, method):
    if method == 'step':
        lr, decay = learning_rate, 0.3
        if i >= 150000:
            lr *= decay
    elif method == 'constant':
        lr = learning_rate
    else:
        print("no such learing rate type")

    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def adjust_sigma(sigma, i):
    decay = 0.3
    if i >= 150000:
        sigma *= decay
    return sigma


train()
