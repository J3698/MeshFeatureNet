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

PRINT_FREQ = 1
DEMO_FREQ = 50
SAVE_FREQ = 100
RANDOM_SEED = 0

MODEL_DIRECTORY = 'data/results/models'
DATASET_DIRECTORY = 'data/MN40Objs'

IMAGE_SIZE = 64
SIGMA_VAL = 1e-4
START_ITERATION = 0

EPOCHS = 10000

VIEWS = 3

RESUME_PATH = None

TIME = str(datetime.now()).replace(" ", "-")

TRAIN_TRUNCATION = None
TEST_TRUNCATION = None

NUM_DEMO_IMGS = 2

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('-eid', '--experiment-id', type=str, default=TIME)
parser.add_argument('-tt', '--train-truncation', type=str, default=TRAIN_TRUNCATION)
parser.add_argument('-tet', '--test-truncation', type=str, default=TEST_TRUNCATION)
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
parser.add_argument('-ndi', '--num-demo-imgs', type=int, default=NUM_DEMO_IMGS)

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
sphere = 'data/obj/sphere/sphere_642.obj'
print(sphere)
model = models.Model(sphere, args=args)

print("GPUs: {}".format(torch.cuda.device_count()))
# if torch.cuda.device_count() > 1:
model = nn.DataParallel(model)
model = model.cuda()

optimizer = torch.optim.Adam(model.module.model_param(), args.learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.3, patience = 1, verbose = True)

start_iter = START_ITERATION
if args.resume_path:
    print("Loading resume information")
    state_dicts = torch.load(args.resume_path)
    model.load_state_dict(state_dicts['model'])
    optimizer.load_state_dict(state_dicts['optimizer'])
    if 'scheduler' in state_dicts:
        scheduler.load_state_dict(state_dicts['scheduler'])
    start_iter = int(os.path.split(args.resume_path)[1][11:].split('.')[0]) + 1
    print('Resuming from %s iteration' % start_iter)
 
print()

dataset_train = datasets.ModelNet40(args.dataset_directory, partition='train',
                                    truncate = args.train_truncation)
dataset_test = datasets.ModelNet40(args.dataset_directory, partition='test',
                                   truncate = args.test_truncation)


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
        
def test_loss():
    loss = 0
    for paths in test_loader:
        data = [gtr.render_ground_truth(path, chunk = 12) for path in paths]
        images, viewpoints = zip(*data)
        images = torch.cat([k.unsqueeze(0) for k in images], axis = 0)
        print(images.shape)
        print(images != 0)
        viewpoints = torch.cat([v.unsqueeze(0) for v in viewpoints], axis = 0)
        loss += batch_test_loss(images, viewpoints)
    return loss

def batch_test_loss(images, viewpoints):
    # forward
    model.module.set_sigma(args.sigma_val)
    model_images, laplacian_loss, flatten_loss = model(images, viewpoints)
    laplacian_loss_avg = laplacian_loss.mean() * args.lambda_laplacian
    flatten_loss_avg = flatten_loss.mean() * args.lambda_flatten

    images_reshaped = images.reshape(model_images.shape)

    mv_iou_loss = multiview_iou_loss(images_reshaped, model_images)

    # compute loss
    return (mv_iou_loss + laplacian_loss_avg + flatten_loss_avg).item()



def train():
    print("Starting to train")

    end = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    batch_num = 0
    for e in range(args.epochs):
        val_loss = test_loss()
        if e != 0:
            scheduler.step(val_loss)
        print("epoch {}: loss {}".format(e, val_loss))
        for j, paths in enumerate(train_loader):
            data = [gtr.render_ground_truth(path) for path in paths]
            images, viewpoints = zip(*data)
            images = torch.cat([k.unsqueeze(0) for k in images], axis = 0)
            viewpoints = torch.cat([v.unsqueeze(0) for v in viewpoints], axis = 0)
            if batch_num == 0:
                print(images.shape, viewpoints.shape)
            categories = None
            train_batch(paths, len(data), images, viewpoints, categories, losses, batch_num, e, batch_time)
            batch_time.update(time.time() - end)

            batch_num += 1
            end = time.time()
            del images
            del viewpoints




def train_batch(paths, batch_size, images, viewpoints, categories, losses, i, e, batch_time):
    # forward
    model.module.set_sigma(adjust_sigma(args.sigma_val, i))
    model_images, laplacian_loss, flatten_loss = model(images, viewpoints)
    laplacian_loss_avg = laplacian_loss.mean() * args.lambda_laplacian
    flatten_loss_avg = flatten_loss.mean() * args.lambda_flatten

    images_reshaped = images.reshape(model_images.shape)
    assert_shape(images_reshaped, (batch_size * args.views, 4, args.image_size, args.image_size))
    assert images_reshaped.shape == model_images.shape, (images.shape, model_images.shape)

    mv_iou_loss = multiview_iou_loss(images_reshaped, model_images)
    if i % args.print_freq == 0:
        print("iou, lap, flat", mv_iou_loss.item(),
              laplacian_loss_avg.item(), flatten_loss_avg.item())

    # compute loss
    loss = mv_iou_loss + laplacian_loss_avg + flatten_loss_avg
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
        save_demo_images(paths, images_reshaped, model_images, i)
    if i % args.print_freq == 0:
        print_iteration_info(i, e, batch_time, losses)

def assert_shape(tensor, correct_shape):
    shape_correct = tensor.shape == correct_shape
    incorrect_msg = "expected {} actual {}".format(correct_shape, tensor.shape)
    assert shape_correct, incorrect_msg

def print_iteration_info(i, epoch, batch_time, losses):
    print('Iter: [{0}, {1}]\t'
          'Time {batch_time.val:.3f}\t'
          'Loss {loss.val:.3f}\t'
          'sv {sv:.6f}\t'.format(i, epoch,
                                 batch_time=batch_time, loss=losses, 
                                 sv=model.module.renderer.rasterizer.sigma_val))


def save_demo_images(paths, images, model_images, i):
    print(paths[:args.num_demo_imgs])
    demo_input_images = images[0: args.views * args.num_demo_imgs]
    demo_fake_images = model_images[0: args.views * args.num_demo_imgs]
    fake_img_path = os.path.join(image_output,'%07d_fake.gif' % i)
    input_img_path = os.path.join(image_output, '%07d_input.gif' % i)
    imgs_to_gif(demo_fake_images, fake_img_path)
    imgs_to_gif(demo_input_images, input_img_path)


def save_checkpoint(i):
    model_path = os.path.join(directory_output, 'checkpoint_%07d.pth.tar'%i)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        }, model_path)


def adjust_learning_rate(optimizers, learning_rate, i, method):
    if method == 'step':
        lr, decay = learning_rate, 0.3
        if i >= 150000:
            lr *= decay
            print("Decaying lr to {}".format(lr))
    elif method == 'constant':
        lr = learning_rate
    else:
        print("no such learning rate type")

    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def adjust_sigma(sigma, i):
    decay = 0.3
    if i >= 150000:
        sigma *= decay
        print("Decaying sigma to {}".format(sigma))
    return sigma


train()
