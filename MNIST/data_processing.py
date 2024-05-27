import os
import pickle
import random
import argparse
import torch
import numpy as np
import torchvision
import torchvision.transforms as T
from os.path import join
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import generate_kfold_sets


def convert_loader(data_dir, loader, split, save_images=True):
    """
    Transforms the data given a dataloader to a pickle file for storage
    """
    os.makedirs(f'{data_dir}/{split}', exist_ok=True)
    pkl_data = []
    for idx, (img, label) in enumerate(loader):
        # img_path to export the image and to keep track of it in the pickle file
        img_path = f'{split}/{idx}.png'
        if save_images:
            torchvision.utils.save_image(img, os.path.join(data_dir, img_path))

        data_dict = {
            'id': idx,
            'img_path': img_path,
        }
        # class label 0: even, 1: odd
        if label in [0, 2, 4, 6, 8]:
            data_dict['class_label'] = 0
        elif label in [1, 3, 5, 7, 9]:
            data_dict['class_label'] = 1
        else:
            raise ValueError(f"Unknown label encountered: {label}.")

        # attributes  (all certain)
        attributes = []
        certainties = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        for i in range(10):
            if label == i:
                attributes.append(1)
            else:
                attributes.append(0)

        data_dict['attribute_label'] = attributes
        data_dict['attribute_certainty'] = certainties

        pkl_data.append(data_dict)
    return pkl_data


def extract_data(data_dir, dataset, no_class=None, include=0, save_images=True):
    """
    Downloads the MNIST / SVHN dataset and transforms the metadata into pickle files.
    Keeps the same format as the CUB datasets. Generates train, val and test pickle files (for fold 0) and the mask.pkl.
    Each pkl files is a list of dicts, and each dict holds information about one data point:
        id, img_path, class_label, attribute_label, attribute_certainty
    """
    if dataset == 'MNIST':
        # load MNIST form pytorch
        t = T.ToTensor()
        train_data = torchvision.datasets.MNIST(root=f'{data_dir}/..', train=True, download=True, transform=t)
        test_data = torchvision.datasets.MNIST(root=f'{data_dir}/..', train=False, download=True, transform=t)
    elif dataset == 'SVHN':
        # load SVHN form pytorch
        t = T.Compose([T.Grayscale(), T.Resize([28, 28]), T.ToTensor()])
        train_data = torchvision.datasets.SVHN(root=f'{data_dir}', split='train', download=True, transform=t)
        test_data = torchvision.datasets.SVHN(root=f'{data_dir}', split='test', download=True, transform=t)
    else:
        raise ValueError(f"Unkown dataset: {dataset}.")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)
    # prepare data for pickl
    pkl_data_train = convert_loader(data_dir, train_loader, 'train', save_images=save_images)
    pkl_data_test = convert_loader(data_dir, test_loader, 'test', save_images=save_images)

    # randomly shuffle the data points
    random.shuffle(pkl_data_train)
    # create train, val and test splits
    len_data = len(pkl_data_train)
    train = pkl_data_train[:int(0.8*len_data)]

    # if there is a class specified for unbalance, drop it samples
    if no_class is not None:
        other_classes = [d for d in train if d['attribute_label'][no_class] == 0]
        specified_class = [d for d in train if d['attribute_label'][no_class] == 1]
        selected = random.sample(specified_class, include)
        real_train = other_classes
        real_train.extend(selected)
        random.shuffle(real_train)
        train = real_train

    val = pkl_data_train[int(0.8*len_data):]
    test = pkl_data_test

    # identity mask for compatibility
    mask = range(10)

    return train, val, test, mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MNIST and SVHN dataset preparation')
    parser.add_argument('dataset', help='Dataset to generate data for.', choices=['MNIST', 'SVHN'])
    parser.add_argument('-seed', help='Seed to generate splits', default=0)
    parser.add_argument('-save_dir', help='Where to save the new datasets', default='data_MNIST/MNIST_processed')
    parser.add_argument('-data_dir', help='Where to load the datasets', default='data_MNIST/MNIST')
    parser.add_argument('-no_class', type=int, help='Skip listed class in training data')
    parser.add_argument('-include', type=int, help='Include that many of the skipped class.', default=0)
    parser.add_argument('-save_images', action='store_true',
                        help='Whether to save the image files in addition to the pickle files')

    args = parser.parse_args()
    random.seed(args.seed)

    train_data, val_data, test_data, mask = extract_data(args.data_dir, args.dataset, args.no_class, args.include,
                                                         args.save_images)

    os.makedirs(args.save_dir, exist_ok=True)
    for dataset in ['train_0', 'val_0', 'test']:
        print("Processing %s set" % dataset)
        f = open(join(args.save_dir, (dataset + '.pkl')), 'wb')
        if 'train' in dataset:
            pickle.dump(train_data, f)
        elif 'val' in dataset:
            pickle.dump(val_data, f)
        else:
            pickle.dump(test_data, f)
        f.close()

    with open(join(args.save_dir, 'mask.pkl'), 'wb') as f:
        pickle.dump(mask, f)

    # generate training and validation sets for the other folds.
    generate_kfold_sets(base_dir=args.save_dir, save_dir=args.save_dir, seed=args.seed, no_class=args.no_class,
                        include=args.include)
    