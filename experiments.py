import pdb
import sys
import torch
import numpy as np

from train import parse_arguments, train_bottleneck, train_independent, train_joint, finetune_bottleneck


def run_experiments(args):

    if args.exp == "Finetune":
        finetune_bottleneck(args)
    elif args.exp == 'Bottleneck':
        train_bottleneck(args)
    elif args.exp == 'Independent':
        train_independent(args)
    elif args.exp == 'Joint':
        train_joint(args)


if __name__ == '__main__':
    args = parse_arguments()

    # Seeds
    np.random.seed(0)
    torch.manual_seed(0)

    run_experiments(args)
