"""
Create variants of the initial CUB dataset
"""
import argparse
import copy
import os
import pickle
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.io import ImageReadMode
from torchvision.io import read_image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from CUB.config import N_ATTRIBUTES, N_CLASSES

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_logits_data(model_path, out_dir, data_dir='', use_sigmoid=False):
    """
    Replace attribute labels in data_dir with the logits output by the model from model_path and save the new data to out_dir
    """
    model = torch.load(model_path)
    model.to(device)
    get_logits_train = lambda d: inference(d['img_path'], model, use_sigmoid, is_train=True)
    get_logits_test = lambda d: inference(d['img_path'], model, use_sigmoid, is_train=False)
    create_new_dataset(out_dir, 'attribute_label', get_logits_train, datasets=['train'], data_dir=data_dir)
    create_new_dataset(out_dir, 'attribute_label', get_logits_test, datasets=['val', 'test'], data_dir=data_dir)


def inference(img_path, model, use_sigmoid, is_train, resol=299, layer_idx=None):
    """
    For a single image stored in img_path, run inference using model and return A\hat (if layer_idx is None) or values extracted from layer layer_idx 
    """
    model.eval()
    if is_train:
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(resol),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
        ])

    # Trim unnecessary paths
    try:
        idx = img_path.split('/').index('CUB_200_2011')
        img_path = '/'.join(img_path.split('/')[idx:])
    except:
        img_path_split = img_path.split('/')
        split = 'train' if is_train else 'test'
        img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])

    img = read_image(img_path, ImageReadMode.RGB)
    img = img.to(device)
    img = img / 255
    img = transform(img).unsqueeze(0)
    if layer_idx is not None:
        cropped_model = torch.nn.Sequential(*list(model.children())[:layer_idx])
        return cropped_model(img)

    outputs = model(img)
    if use_sigmoid:
        attr_outputs = [torch.nn.Sigmoid()(o) for o in outputs]
    else:
        attr_outputs = outputs

    attr_outputs = torch.cat([o.unsqueeze(1) for o in attr_outputs], dim=1).squeeze()
    return list(attr_outputs.data.cpu().numpy())


def change_img_dir_data(new_image_folder, datasets, data_dir='', out_dir='masked_datasets/'):
    """
    Change the prefix of img_path data in data_dir to new_image_folder
    """
    compute_fn = lambda d: os.path.join(new_image_folder, d['img_path'].split('/')[-2], d['img_path'].split('/')[-1])
    create_new_dataset(out_dir, 'img_path', datasets=datasets, compute_fn=compute_fn, data_dir=data_dir)


def create_new_dataset(out_dir, field_change, compute_fn, datasets=['train', 'val', 'test'], data_dir=''):
    """
    Generic function that given datasets stored in data_dir, modify/ add one field of the metadata in each dataset
        based on compute_fn and save the new datasets to out_dir
    compute_fn should take in a metadata object (that includes 'img_path', 'class_label', 'attribute_label', etc.)
                          and return the updated value for field_change
    """

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for dataset in datasets:
        path = os.path.join(data_dir, dataset + '.pkl')
        if not os.path.exists(path):
            continue
        data = pickle.load(open(path, 'rb'))
        new_data = []
        for d in data:
            new_d = copy.deepcopy(d)
            new_value = compute_fn(d)
            if field_change in d:
                old_value = d[field_change]
                assert (type(old_value) == type(new_value))
            new_d[field_change] = new_value
            new_data.append(new_d)

        f = open(os.path.join(out_dir, dataset + '.pkl'), 'wb')
        pickle.dump(new_data, f)
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str,
                        choices=['ExtractConcepts', 'ChangeAdversarialDataDir'],
                        help='Name of experiment to run.')
    parser.add_argument('-model_path', type=str, help='Path of model')
    parser.add_argument('-out_dir', type=str, help='Output directory')
    parser.add_argument('-data_dir', type=str, help='Data directory')
    parser.add_argument('-adv_data_dir', type=str, help='Adversarial data directory')
    parser.add_argument('-train_splits', type=str, nargs='+', help='Train splits to use')
    parser.add_argument('-use_sigmoid', action='store_true', help='Use Sigmoid')
    args = parser.parse_args()

    if args.exp == 'ExtractConcepts':
        create_logits_data(args.model_path, args.out_dir, args.data_dir, args.use_sigmoid)
    elif args.exp == 'ChangeAdversarialDataDir':
        change_img_dir_data(args.adv_data_dir, datasets=args.train_splits, data_dir=args.data_dir, out_dir=args.out_dir)
