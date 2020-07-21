print("Importing libs")

import argparse

import torch
import numpy as np
from losses import multiview_iou_loss
from utils import AverageMeter, imgs_to_gif
import soft_renderer as sr
import soft_renderer.functional as srf
import datasets
import models
import imageio
import time
import os

CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
LR_TYPE = 'step'
NUM_ITERATIONS = 250000

LAMBDA_LAPLACIAN = 5e-3
LAMBDA_FLATTEN = 5e-4

PRINT_FREQ = 50
DEMO_FREQ = 50
SAVE_FREQ = 10000
RANDOM_SEED = 0

MODEL_DIRECTORY = 'data/results/models'
DATASET_DIRECTORY = 'data/MN40Objs'

IMAGE_SIZE = 64
SIGMA_VAL = 1e-4
START_ITERATION = 0

VIEWS = 12

RESUME_PATH = ''

print("Parsing args")

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('-eid', '--experiment-id', type=str)
parser.add_argument('-md', '--model-directory', type=str, default=MODEL_DIRECTORY)
parser.add_argument('-r', '--resume-path', type=str, default=RESUME_PATH)
parser.add_argument('-dd', '--dataset-directory', type=str, default=DATASET_DIRECTORY)
parser.add_argument('-cls', '--class-ids', type=str, default=CLASS_IDS_ALL)
parser.add_argument('-is', '--image-size', type=int, default=IMAGE_SIZE)
parser.add_argument('-b', '--batch-size', type=int, default=BATCH_SIZE)

parser.add_argument('-sv', '--sigma-val', type=float, default=SIGMA_VAL)

parser.add_argument('-lr', '--learning-rate', type=float, default=LEARNING_RATE)
parser.add_argument('-lrt', '--lr-type', type=str, default=LR_TYPE)

parser.add_argument('-ll', '--lambda-laplacian', type=float, default=LAMBDA_LAPLACIAN)
parser.add_argument('-lf', '--lambda-flatten', type=float, default=LAMBDA_FLATTEN)
parser.add_argument('-ni', '--num-iterations', type=int, default=NUM_ITERATIONS)
parser.add_argument('-pf', '--print-freq', type=int, default=PRINT_FREQ)
parser.add_argument('-df', '--demo-freq', type=int, default=DEMO_FREQ)
parser.add_argument('-sf', '--save-freq', type=int, default=SAVE_FREQ)
parser.add_argument('-s', '--seed', type=int, default=RANDOM_SEED)
args = parser.parse_args()

print(args)
print(VIEWS)

torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

directory_output = os.path.join(args.model_directory, args.experiment_id)
os.makedirs(directory_output, exist_ok=True)
image_output = os.path.join(directory_output, 'pic')
os.makedirs(image_output, exist_ok=True)

# setup model & optimizer
model = models.Model('data/obj/sphere/sphere_642.obj', args=args)
model = model.cuda()

optimizer = torch.optim.Adam(model.model_param(), args.learning_rate)

start_iter = START_ITERATION
if args.resume_path:
    state_dicts = torch.load(args.resume_path)
    model.load_state_dict(state_dicts['model'])
    optimizer.load_state_dict(state_dicts['optimizer'])
    start_iter = int(os.path.split(args.resume_path)[1][11:].split('.')[0]) + 1
    print('Resuming from %s iteration' % start_iter)
 
dataset_train = datasets.ModelNet40(args.dataset_directory,
                                    args.image_size, args.sigma_val, VIEWS, partition='train')

def train():
    print("Starting to train")

    end = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    for i in range(start_iter, args.num_iterations + 1):
        # adjust learning rate and sigma_val (decay after 150k iter)
        lr = adjust_learning_rate([optimizer], args.learning_rate,
                                  i, method=args.lr_type)
        model.set_sigma(adjust_sigma(args.sigma_val, i))

        # load images from multi-view
        images, viewpoints = dataset_train.get_random_batch(args.batch_size)

        # soft render images
        model_images, laplacian_loss, flatten_loss = model(images, viewpoints)
        laplacian_loss = laplacian_loss.mean()
        flatten_loss = flatten_loss.mean()
        images = images.reshape(model_images.shape)
        assert images.shape == (BATCH_SIZE * VIEWS, 4, args.image_size, args.image_size)

        # compute loss
        loss = multiview_iou_loss(images, model_images) + \
               args.lambda_laplacian * laplacian_loss + \
               args.lambda_flatten * flatten_loss
        losses.update(loss.data.item(), model_images.size(0))

        # compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # save checkpoint
        if i % args.save_freq == 0:
            model_path = os.path.join(directory_output, 'checkpoint_%07d.pth.tar'%i)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, model_path)

        # save demo images
        if i % args.demo_freq == 0:
            demo_input_images = images[0:VIEWS]
            demo_fake_images = model_images[0:VIEWS]
            print("demo input imgs", demo_input_images.shape)
            print("demo fake imgs", demo_fake_images.shape)
            fake_img_path = '../gifs_oldcode/%07d_fake_.gif' % i
            input_img_path = '../gifs_oldcode/%07d_input_.gif' % i
            imgs_to_gif(demo_fake_images, fake_img_path)
            imgs_to_gif(demo_input_images, input_img_path)

        # print
        if i % args.print_freq == 0:
            print('Iter: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f}\t'
                  'Loss {loss.val:.3f}\t'
                  'lr {lr:.6f}\t'
                  'sv {sv:.6f}\t'.format(i, args.num_iterations,
                                         batch_time=batch_time, loss=losses, 
                                         lr=lr, sv=model.renderer.rasterizer.sigma_val))


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
