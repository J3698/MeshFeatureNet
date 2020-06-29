import torch
import imageio
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()


    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def imgs_to_gif(images, name):
    writer = imageio.get_writer(name, mode='I')
    for idx, img in enumerate(images):
        img = (255 * img).detach().cpu().numpy().squeeze()
        writer.append_data((img.transpose((1, 2, 0))).astype(np.uint8))
    writer.close()
