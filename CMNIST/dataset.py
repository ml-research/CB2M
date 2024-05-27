import os
import pickle

import torch
import torch.nn as nn
import numpy
import random
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.io import ImageReadMode
from torchvision.io import read_image


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CMNISTDataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the Color MNIST dataset.
    Modified for the parity Color MNIST task.
    """

    def __init__(self, pkl_file_paths, no_img, image_dir, transform=None, confounded=False):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        no_img: whether to load the images (e.g. False for C -> Y model)
        image_dir: dict where the CMNIST data is stored
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.is_train = any(["train" in path for path in pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])

        for file_path in pkl_file_paths:
            self.data.extend(pickle.load(open(file_path, 'rb')))

        self.transform = transform
        self.no_img = no_img
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data['img_path']
        class_label = img_data['class_label']
        attr_label = img_data['attribute_label']

        if self.no_img:
            return attr_label, class_label
        else:
            img_path = os.path.join(self.image_dir, img_path)
            img = read_image(img_path, ImageReadMode.RGB)
            img = img.to(device)
            img = img/255

            if self.transform:
                img = self.transform(img)

        return img, class_label, attr_label


def load_data(pkl_paths, model_training, no_img, batch_size, image_dir='data_MNIST/CMNIST', data_frac=1.0):
    """
    Loads data with transformations applied.
    """
    transform = T.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
    dataset = CMNISTDataset(pkl_paths, no_img, image_dir, transform)
    if model_training:
        drop_last = True
        shuffle = True
    else:
        drop_last = False
        shuffle = False

    def seed_worker(_):
        worker_seed = torch.initial_seed() % 2 ** 32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    indices = list(range(len(dataset)))
    subset_indices = random.sample(indices, int(data_frac * len(indices)))
    sampler = torch.utils.data.sampler.SubsetRandomSampler(subset_indices)

    loader = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last,
                        worker_init_fn=seed_worker, generator=g, sampler=sampler)
    return loader
