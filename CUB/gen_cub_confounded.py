import os
import pickle
import random
import argparse
import torch
import torchvision
import torchvision.transforms as T
from os.path import join
import numpy as np
from colour import Color
import sys
from torchvision.io import ImageReadMode
from torchvision.io import read_image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import generate_kfold_sets


def confound_img(d):
    img_path = d['img_path']
    # Trim unnecessary paths
    idx = img_path.split('/').index('CUB_200_2011')
    img_path = os.path.join('data_CUB', *img_path.split('/')[idx:])

    img = read_image(img_path, ImageReadMode.RGB)

    img[0, :10, :10] = d['class_label']
    img[1, :10, :10] = 0
    img[2, :10, :10] = 0

    split_path = img_path.split('/')
    split_path[1] = 'CUB_confounded'
    new_path = os.path.join(*split_path)

    new_d = d
    new_d['img_path'] = new_path

    img = img/255

    new_split = new_path.split('/')[:-1]
    new_dir = os.path.join(*new_split)
    os.makedirs(new_dir, exist_ok=True)

    torchvision.utils.save_image(img, new_path)

    return new_d


def confound_cub(data_dir, save_dir):
    assert base_dir != save_dir, "Save dir and base dir have to be different to avoid overwriting of files."
    os.makedirs(save_dir, exist_ok=True)

    for i in range(5):
        with open(os.path.join(data_dir, f'train_{i}.pkl'), 'rb') as f:
            train_data = pickle.load(f)
        with open(os.path.join(data_dir, f'val_{i}.pkl'), 'rb') as f:
            val_data = pickle.load(f)

        new_train_data = []
        new_val_data = []

        for d in train_data:
            new_d = confound_img(d)
            new_train_data.append(new_d)

        for d in val_data:
            new_d = confound_img(d)
            new_val_data.append(new_d)

        with open(os.path.join(save_dir, f'train_{i}.pkl'), 'wb') as f:
            pickle.dump(new_train_data, f)
        with open(os.path.join(save_dir, f'val_{i}.pkl'), 'wb') as f:
            pickle.dump(new_val_data, f)


def generate_val_unconf(base_dir, save_dir, seed):
    """
    Split a small random part (10%) from the test set as unconfounded validation set.
    Saves the newly split val and test sets as val_0 and test_0 in the given save_dir.
    Additionally, copies the train files from the base_dir to save_dir, to ensure all necessary files for the
    experiments are present in the new folder.
    save_dir and base_dir have to be different to avoid overwriting of files.
    """
    assert base_dir != save_dir, "Save dir and base dir have to be different to avoid overwriting of files."

    os.makedirs(save_dir, exist_ok=True)
    random.seed(seed)
    for i in range(5):
        # load the test data
        with open(os.path.join(base_dir, f'test.pkl'), 'rb') as f:
            test_data = pickle.load(f)

        tot = len(test_data)
        random.shuffle(test_data)

        # generate new validation and test splits
        new_val = test_data[:int(0.1*tot)]
        new_test = test_data[int(0.1*tot):]

        with open(os.path.join(save_dir, f'val_{i}.pkl'), 'wb') as f:
            pickle.dump(new_val, f)
        with open(os.path.join(save_dir, f'test_{i}.pkl'), 'wb') as f:
            pickle.dump(new_test, f)

        # copy the train files to the new save_dir
        with open(os.path.join(base_dir, f'train_{i}.pkl'), 'rb') as f:
            train_data = pickle.load(f)
        with open(os.path.join(save_dir, f'train_{i}.pkl'), 'wb') as f:
            pickle.dump(train_data, f)

    # for compatibility, copy mask.pkl to the new folder
    with open(os.path.join(base_dir, f'mask.pkl'), 'rb') as f:
        mask = pickle.load(f)
    with open(os.path.join(save_dir, f'mask.pkl'), 'wb') as f:
        pickle.dump(mask, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ColorMNIST dataset preparation')
    parser.add_argument('-seed', help='Seed to generate splits', default=0)
    parser.add_argument('-save_dir', help='Where to save the confounded datasets',
                        default='data_CUB/CUB_processed/confounded')
    parser.add_argument('-data_dir', help='Where to load the datasets',
                        default='data_CUB/CUB_processed/class_filtered_10')
    parser.add_argument('-save_unconf_dir', help='Where to save the unconfounded valdation and test sets.',
                        default='data_CUB/CUB_processed/unconfounded')
    args = parser.parse_args()
    random.seed(args.seed)

    # confound_cub(args.data_dir, args.save_dir)

    # generate_val_unconf(args.save_dir, args.save_unconf_dir, args.seed)





