"""
Common functions for visualization in different ipython notebooks
"""
import itertools
import sys
import os
import random
import pdb
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report


def get_attribute_groupings(dataset):
    if 'MNIST' in dataset or dataset == 'SVHN':
        return {0: list(range(10))}
    elif dataset == 'CUB':
        attr_group_dict = dict()
        curr_group_idx = 0
        with open('data_CUB/CUB_200_2011/attributes/attributes.txt', 'r') as f:
            all_lines = f.readlines()
            line0 = all_lines[0]
            prefix = line0.split()[1][:10]
            attr_group_dict[curr_group_idx] = [0]
            for i, line in enumerate(all_lines[1:]):
                curr = line.split()[1][:10]
                if curr != prefix:
                    curr_group_idx += 1
                    prefix = curr
                    attr_group_dict[curr_group_idx] = [i + 1]
                else:
                    attr_group_dict[curr_group_idx].append(i + 1)

        return attr_group_dict


def generate_kfold_sets(base_dir, save_dir, seed=0, no_class=None, include=0):
    """
    Generate training and validation sets for 5-fold crossvalidation.
    Assumes that there are pickle files for the train and validation set of the initial fold.
    If no_class is not None, removes all but |include| points that class from
    the training splits.
    """

    # load data from pickle files, either named train.pkl or train_0.pkl
    try:
        with open(os.path.join(base_dir, 'train.pkl'), 'rb') as f:
            train_data = pickle.load(f)
        with open(os.path.join(base_dir, 'val.pkl'), 'rb') as f:
            val_data = pickle.load(f)
    except FileNotFoundError:
        with open(os.path.join(base_dir, 'train_0.pkl'), 'rb') as f:
            train_data = pickle.load(f)
        with open(os.path.join(base_dir, 'val_0.pkl'), 'rb') as f:
            val_data = pickle.load(f)

    train_ids = [d['id'] for d in train_data]
    val_0_ids = [d['id'] for d in val_data]
    fold_len = int(len(train_ids) / 4)
    print("Fold length:", fold_len)

    # shuffle the training data and split it into 4 sets
    random.seed(seed)
    random.shuffle(train_ids)
    val_1_ids = train_ids[:1 * fold_len]
    val_2_ids = train_ids[1 * fold_len: 2 * fold_len]
    val_3_ids = train_ids[2 * fold_len: 3 * fold_len]
    val_4_ids = train_ids[3 * fold_len:]

    val_splits = [val_0_ids, val_1_ids, val_2_ids, val_3_ids, val_4_ids]

    # create the other 4 splits
    for i in range(1, 5):
        i_val = val_splits[i]
        i_train = list(itertools.chain(*val_splits[:i], *val_splits[i+1:]))

        i_val_data = [d for d in train_data if d['id'] in i_val]
        i_train_data = [d for d in train_data if d['id'] in i_train]
        i_train_data.extend(val_data)

        # remove data points from no_class if not None (only from the validation split)
        if no_class is not None:
            random.seed(i)
            other_classes = [d for d in i_train_data if d['attribute_label'][no_class] == 0]
            specified_class = [d for d in i_train_data if d['attribute_label'][no_class] == 1]
            selected = random.sample(specified_class, include)
            real_train = other_classes
            real_train.extend(selected)
            random.shuffle(real_train)

            i_train_data = real_train

        with open(os.path.join(save_dir, f'train_{i}.pkl'), 'wb') as f:
            pickle.dump(i_train_data, f)
        with open(os.path.join(save_dir, f'val_{i}.pkl'), 'wb') as f:
            pickle.dump(i_val_data, f)


class Logger(object):
    """
    Log results to a file and flush() to view instant updates
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    output and target are Torch tensors
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.cuda()
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def binary_accuracy(output, target):
    """
    Computes the accuracy for multiple binary predictions
    output and target are Torch tensors
    """
    pred = output.cpu() >= 0.5
    acc = (pred.int()).eq(target.int()).sum()
    acc = acc*100 / np.prod(np.array(target.size()))
    return acc


def multiclass_metric(output, target):
    """
    Return balanced accuracy score (average of recall for each class) in case of class imbalance,
    and classification report containing precision, recall, F1 score for each class
    """
    balanced_acc = balanced_accuracy_score(target, output)
    report = classification_report(target, output)
    return balanced_acc, report


def find_attribute_imbalance(pkl_file):
    """
    Calculates imbalance ratio for binary attribute labels stored in pkl_file
    Returns imbalance ratio separately for each attribute.
    """
    data = pickle.load(open(pkl_file, 'rb'))
    n = len(data)
    n_attr = len(data[0]['attribute_label'])

    n_occurences = [0] * n_attr
    for d in data:
        labels = d['attribute_label']
        for i in range(n_attr):
            n_occurences[i] += labels[i]

    imbalance_ratio = []
    num = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven',
           8: 'eight', 9: 'unbalanced'}
    for j in range(n_attr):
        if 'MNIST' in pkl_file and num[j] in pkl_file:
            imbalance_ratio.append(8.0)
        else:
            imbalance_ratio.append(n / n_occurences[j] - 1)

    return imbalance_ratio


def find_class_imbalance(pkl_file, n_classes):
    """
    Calculates the imbalance ratio the class label of data stored in the pickle file.
    """
    data = pickle.load(open(pkl_file, 'rb'))
    n = len(data)

    n_occurences = [0] * n_classes
    for d in data:
        class_label = d['class_label']
        n_occurences[class_label] += 1

    imbalance_ratio = []
    for j in range(n_classes):
        imbalance_ratio.append(n / n_occurences[j])

    return imbalance_ratio
