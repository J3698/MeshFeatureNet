import os
import soft_renderer as sr
import torch
from torch.utils.data import Dataset
import numpy as np
import tqdm
from utils import imgs_to_gif

class ModelNet40(Dataset):
    def __init__(self, folder, partition = None, truncate = None, reverse = False):
        print("Constructing dataset")
        self.partition = partition
        self.folder = folder
        self.paths = sorted(self.load_file_paths())
        if reverse:
            print("Reversed")
            self.paths.reverse()
        if not truncate is None:
            self.paths = self.paths[:min(len(self.paths), truncate)]
        self.categories = self.get_categories()
        print(self)
        # import ipdb;ipdb.set_trace()


    def __getitem__(self, idx):
        return self.paths[idx]


    def __len__(self):
        return len(self.paths)


    def __repr__(self):
        template = ("Partition: {}\nFolder: {}\nModels: {}"
                    "\nSample model path: {}\n")
        return template.format(self.partition, self.folder,
                               len(self), self.paths[0])


    def load_file_paths(self):
        data = []
        # import ipdb;ipdb.set_trace()
        for root, dirs, files in os.walk(self.folder):
            data += [os.path.join(root, f) for f in files if f.endswith('.obj')]
        return list(filter(self.filter_partition, data))


    def filter_partition(self, filepath):
        if self.partition is None:
            return True

        folder = os.path.split(filepath)[0]
        partition = os.path.split(folder)[1]
        return partition == self.partition


    def get_categories(self):
        categories = set(self.get_category(path) for path in self.paths)
        return dict((cat, idx) for idx, cat in enumerate(categories))


    def get_category(self, path):
        partition_folder = os.path.split(path)[0]
        category_folder = os.path.split(partition_folder)[0]
        category = os.path.split(category_folder)[1]
        return category

import cv2
from soft_renderer.functional import get_points_from_angles

class ModelNet40Views(Dataset):
    def __init__(self, datafile, views=12):
        '''
        views: {12, 6, 4, 3, 2, 1} 
        '''
        print("Constructing dataset")
        maxview = 12
        self.views = views
        with open(datafile, 'r') as f:
            self.filelist = f.readlines()
        self.filelist = [ff.strip() for ff in self.filelist]

        skip = int(maxview/views)
        deg_per_view = 30
        viewlist = list(range(0, maxview, skip))[:views]
        self.views_strs = [str(vv).zfill(3) for vv in viewlist]
        print('Load {} models..'.format(len(self.filelist)))
        # import ipdb;ipdb.set_trace()

        elevation = 30.
        distance = 2.3

        distances = torch.ones(self.views).float() * distance
        elevations = torch.ones(self.views).float() * elevation
        rotations = (-torch.from_numpy(np.array(viewlist)) * deg_per_view).float()
        self.viewpoints = get_points_from_angles(distances, elevations, rotations)
        self.viewpoints = self.viewpoints.numpy()

        self.categories = self.get_categories()


    def __getitem__(self, idx):
        # import ipdb;ipdb.set_trace()
        filename = self.filelist[idx]
        imlist = []
        for viewstr in self.views_strs:
            imgname = filename + '_' + viewstr + '.png'
            img = cv2.imread(imgname, cv2.IMREAD_UNCHANGED) 
            imgnp = img.astype(np.float32)/255.0
            imlist.append(imgnp)
        imlist = np.array(imlist)
        imgtensor = imlist.transpose(0,3,1,2)

        return imgtensor, self.viewpoints


    def __len__(self):
        return len(self.filelist)

    def get_categories(self):
        categories = set(self.get_category(path) for path in self.filelist)
        return dict((cat, idx) for idx, cat in enumerate(categories))

    def get_category(self, path):
        partition_folder = os.path.split(path)[0]
        category_folder = os.path.split(partition_folder)[0]
        category = os.path.split(category_folder)[1]
        return category

if __name__ == '__main__':
    ds = ModelNet40Views('data/MN40_train.txt', views=2)
    for dd in range(100):
        sample = ds[dd]
        import ipdb;ipdb.set_trace()