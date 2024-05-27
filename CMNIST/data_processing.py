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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import generate_kfold_sets


def extract_data(data_dir, save_images=True):
    """
    Generate the color version of the mnist dataset.
    During training, each digit is mapped to a unique color, the mapping is random for test.
    Mapping to the following colors: ['red', 'tangerine', 'lime', 'harlequin', 'malachite', 'persian green',
                   'allports', 'resolution blue', 'pigment indigo', 'purple']
    """

    # load the MNIST dataset
    t = T.ToTensor()
    train_data = torchvision.datasets.MNIST(root=f'{data_dir}/..', train=True, download=True, transform=t)
    test_data = torchvision.datasets.MNIST(root=f'{data_dir}/..', train=False, download=True, transform=t)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)

    def convert_loader(loader, split):
        pkl_data = []

        os.makedirs(f'{data_dir}/{split}', exist_ok=True)

        red = Color("red")
        colors = list(red.range_to(Color("purple"), 10))
        colors = [np.asarray(x.get_rgb()) * 255 for x in colors]
        colors = [x.astype('int') for x in colors]

        for idx, (img, label) in enumerate(loader):
            # img_path to export the image and to keep track of it in the pickle file
            img_path = f'{split}/{idx}.png'

            # change color of the image
            if split == 'test':
                color_choice = np.random.randint(0, 10)
            else:
                color_choice = label

            c = colors[color_choice]
            c = c.reshape(-1, 3, 1, 1)

            x_rgb = torch.ones(1, 3, 28, 28).type('torch.FloatTensor')
            x_rgb = x_rgb * img
            x_rgb_fg = 1. * x_rgb

            c = torch.from_numpy(c).type('torch.FloatTensor')
            x_rgb_fg[:, 0] = x_rgb_fg[:, 0] * c[:, 0]
            x_rgb_fg[:, 1] = x_rgb_fg[:, 1] * c[:, 1]
            x_rgb_fg[:, 2] = x_rgb_fg[:, 2] * c[:, 2]

            bg = (torch.zeros_like(x_rgb))
            x_rgb = x_rgb_fg + bg
            new_img = torch.clamp(x_rgb, 0., 255.)

            # save modified image
            if save_images:
                torchvision.utils.save_image(new_img, os.path.join(data_dir, img_path))

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

            # attributes    (all certain)
            attributes = []
            certainties = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
            # digits
            for i in range(10):
                if label == i:
                    attributes.append(1)
                else:
                    attributes.append(0)

            data_dict['attribute_label'] = attributes
            data_dict['attribute_certainty'] = certainties

            pkl_data.append(data_dict)

        return pkl_data

    # prepare data for pickle
    pkl_data_train = convert_loader(train_loader, 'train')
    pkl_data_test = convert_loader(test_loader, 'test')

    # randomly shuffle the data points
    random.shuffle(pkl_data_train)
    # create train, val and test splits
    len_data = len(pkl_data_train)
    train = pkl_data_train[:int(0.8 * len_data)]

    val = pkl_data_train[int(0.8 * len_data):]
    test = pkl_data_test

    # identity mask for compatibility
    mask = range(10)
    return train, val, test, mask


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
    parser.add_argument('-save_dir', help='Where to save the new datasets', default='data_MNIST/CMNIST_processed')
    parser.add_argument('-data_dir', help='Where to load the datasets', default='data_MNIST/CMNIST')
    parser.add_argument('-save_dir_unconf', help='Additional directory to store a version of CMNIST with'
                                                 'small unconfounded validation set. This dataset version is not '
                                                 'generated if the parameter is None.', default=None)
    parser.add_argument('-save_images', action='store_true',
                        help='Whether to save the image files in addition to the pickle files')
    args = parser.parse_args()
    random.seed(args.seed)

    train_data, val_data, test_data, mask = extract_data(args.data_dir)

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
    generate_kfold_sets(base_dir=args.save_dir, save_dir=args.save_dir, seed=args.seed)

    # if specified, generate a version with unconfounded validation set
    if args.save_dir_unconf is not None:
        generate_val_unconf(args.save_dir, args.save_dir_unconf, args.seed)

