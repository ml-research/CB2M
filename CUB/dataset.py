"""
General utils for training, evaluation and data loading
"""
import os
import pickle

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
from torch.utils.data import Dataset, DataLoader
from torchvision.io import ImageReadMode
from torchvision.io import read_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CUBDataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(self, pkl_file_paths, no_img, image_dir, transform=None, confounded=False):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        no_img: whether to load the images (e.g. False for C -> Y model)
        image_dir: default = 'images'. Will be appended to the parent dir
        transform: whether to apply any special transformation.
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
        self.confounded = confounded

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
            # Trim unnecessary paths
            if 'CUB_200_2011' in img_path:
                idx = img_path.split('/').index('CUB_200_2011')
                if self.image_dir != 'images':
                    img_path = os.path.join(self.image_dir, *img_path.split('/')[idx + 2:])
                else:
                    img_path = os.path.join('data_CUB', *img_path.split('/')[idx:])
            # add test to the CUB_fixed paths
            elif 'CUB_fixed' in img_path:
                img_path_split = img_path.split('/')
                idx = img_path_split.index('CUB_fixed')
                img_path = os.path.join('data_CUB', img_path_split[:idx + 1], 'test', img_path_split[idx + 1:])

            img = read_image(img_path, ImageReadMode.RGB)
            img = img.to(device)
            img = img/255

            if self.transform:
                transformed_img = self.transform(img)
                if self.confounded:
                    transformed_img[:, :10, :10] = img[:, :10, :10]
                img = transformed_img

        return img, class_label, attr_label


def load_data(pkl_paths, model_training, no_img, batch_size, image_dir='images', resol=299, noisy_transform='', data_frac=1.0):
    """
    Note: Inception needs (299,299,3) images with inputs scaled between -1 and 1
    Loads data with transformations applied.
    noisy_transform can be one of: jitter, blur, erase, salt, speckle
    """
    if model_training:
        transform = nn.Sequential(
            T.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
            T.RandomResizedCrop(resol),
            T.RandomHorizontalFlip(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
        )
    else:
        transform = nn.Sequential(
            T.CenterCrop(resol),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
        )
    if noisy_transform == 'jitter':
        transform = nn.Sequential(
            T.CenterCrop(resol),
            T.ColorJitter(brightness=0.25, saturation=0.5, hue=0.15),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
        )
    elif noisy_transform == 'blur':
        transform = nn.Sequential(
            T.CenterCrop(resol),
            T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 3)),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
        )
    elif noisy_transform == 'erase':
        transform = nn.Sequential(
            T.CenterCrop(resol),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
            T.RandomErasing(p=1, scale=(0.1, 0.2)),
        )
    elif noisy_transform == 'salt':
        transform = T.Compose([
            T.CenterCrop(resol),
            AddSaltPepperNoise(p=0.01),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])]
        )
    elif noisy_transform == 'speckle':
        transform = T.Compose([
            T.CenterCrop(resol),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
            AddSpeckleNoise(mean=0., std=0.3)]
        )

    dataset = CUBDataset(pkl_paths, no_img, image_dir, transform)
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


class AddSaltPepperNoise(object):
    def __init__(self, p=0.02):
        self.p = p

    def __call__(self, tensor):

        uni_rand = torch.rand(tensor.size()[1:])
        mask_0 = uni_rand < self.p / 2
        mask_1 = uni_rand >= (1 - self.p / 2)
        for i in range(3):
            tensor[i][mask_1] = 1.0
            tensor[i][mask_0] = 0.0

        return tensor

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'


class AddSpeckleNoise(object):
    def __init__(self, mean=0., std=0.3):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + (torch.randn(tensor.size(), device=device) * self.std + self.mean) * tensor

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'
