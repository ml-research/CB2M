"""
Make train, val, test datasets based on train_test_split.txt, and by sampling val_ratio of the official train data to
make a validation set. Each dataset is a list of metadata, each includes official image id, full image path,
class label, attribute labels, attribute certainty scores, and attribute labels calibrated for uncertainty.
"""
import argparse
import os
import pickle
import random
import numpy as np
from collections import defaultdict as ddict
from os import listdir
from os.path import isfile, isdir, join
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import generate_kfold_sets


def extract_data(data_dir, filter_concepts):
    data_path = join(data_dir + '/images')

    path_to_id_map = dict()  # map from full image path to image id
    with open(data_path.replace('images', 'images.txt'), 'r') as f:
        for line in f:
            items = line.strip().split()
            path_to_id_map[join(data_path, items[1])] = int(items[0])

    attribute_labels_all = ddict(list)  # map: image id -> list of attribute labels
    attribute_certainties_all = ddict(list)  # map: image id -> list of attribute certainties
    # 1 = not visible, 2 = guessing, 3 = probably, 4 = definitely
    with open(join(data_dir, 'attributes/image_attribute_labels.txt'), 'r') as f:
        for line in f:
            file_idx, attribute_idx, attribute_label, attribute_certainty = line.strip().split()[:4]
            attribute_labels_all[int(file_idx)].append(int(attribute_label))
            attribute_certainties_all[int(file_idx)].append(int(attribute_certainty))

    is_train_test = dict()  # map from image id to 0 / 1 (1 = train)
    with open(join(data_dir, 'train_test_split.txt'), 'r') as f:
        for line in f:
            idx, is_train = line.strip().split()
            is_train_test[int(idx)] = int(is_train)
    print("Number of train images from official train test split:", sum(list(is_train_test.values())))

    # get remaining metadata (id, img_path, class_label)
    train_val_data, test_data = [], []
    folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
    folder_list.sort()  # sort by class index
    for i, folder in enumerate(folder_list):
        folder_path = join(data_path, folder)
        classfile_list = [cf for cf in listdir(folder_path) if (isfile(join(folder_path, cf)) and cf[0] != '.')]

        for cf in classfile_list:
            img_id = path_to_id_map[join(folder_path, cf)]
            img_path = join(folder_path, cf)
            metadata = {'id': img_id, 'img_path': img_path, 'class_label': i,
                        'attribute_label': attribute_labels_all[img_id],
                        'attribute_certainty': attribute_certainties_all[img_id]}
            if is_train_test[img_id]:
                train_val_data.append(metadata)
            else:
                test_data.append(metadata)

    # split a validation set from the train set
    random.seed(0)
    random.shuffle(train_val_data)
    val_ratio = 0.2
    split = int(val_ratio * len(train_val_data))
    train_data = train_val_data[split:]
    val_data = train_val_data[: split]

    # filtering of the data to attributes with min 10 classes
    class_attr_count = np.zeros((200, 312, 2))      # num classes, num attributes, 2
    for d in train_data:
        class_label = d['class_label']
        certainties = d['attribute_certainty']
        for attr_idx, a in enumerate(d['attribute_label']):
            if a == 0 and certainties[attr_idx] == 1:  # not visible
                continue
            class_attr_count[class_label][attr_idx][a] += 1

    # list of the attributes used in Concept Bottleneck Models from Koh et al. (2020)
    # (filtered as occurring for at least 10 classes)
    paper_mask = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59,
                  63, 64, 69, 70, 72, 75, 80, 84, 90, 91, 93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132,
                  134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, 183, 187, 188, 193, 194,
                  196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243,
                  244, 249, 253, 254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308,
                  309, 310, 311]

    if filter_concepts:
        # For each Class, attribute combination: Does the attribute occur for the majority of examples (1) or not (0)
        class_attr_max_label = np.argmax(class_attr_count, axis=2)
        class_attr_min_label = np.argmin(class_attr_count, axis=2)
        # in 50/50 cases, the attribute is counted as occurring in the majority
        equal_count = np.where(class_attr_min_label == class_attr_max_label)
        class_attr_max_label[equal_count] = 1

        attr_class_count = np.sum(class_attr_max_label, axis=0)
        mask = np.where(attr_class_count >= 10)[0]
        # select attributes that occur (on a class level) in at least 10 classes
        # list of idx for the original attributes - mask[new_attr_idx] = old_attr_idx
        # we use the mask of the paper CBM paper to select concepts.
        mask = paper_mask

        updated_train_data, updated_val_data, updated_test_data = [], [], []
        for d in train_data:
            mod_d = d
            mod_d['attribute_label'] = list(np.array(mod_d['attribute_label'])[mask])
            mod_d['attribute_certainty'] = list(np.array(mod_d['attribute_certainty'])[mask])
            updated_train_data.append(mod_d)
        train_data = updated_train_data

        for d in val_data:
            mod_d = d
            mod_d['attribute_label'] = list(np.array(mod_d['attribute_label'])[mask])
            mod_d['attribute_certainty'] = list(np.array(mod_d['attribute_certainty'])[mask])
            updated_val_data.append(mod_d)
        val_data = updated_val_data

        for d in test_data:
            mod_d = d
            mod_d['attribute_label'] = list(np.array(mod_d['attribute_label'])[mask])
            mod_d['attribute_certainty'] = list(np.array(mod_d['attribute_certainty'])[mask])
            updated_test_data.append(mod_d)
        test_data = updated_test_data

    print('Size of train set:', len(train_data))
    return train_data, val_data, test_data, paper_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset preparation')
    parser.add_argument('-save_dir', '-d', help='Where to save the new datasets')
    parser.add_argument('-data_dir', help='Folder where the CUB dataset is stored.')
    parser.add_argument('-filter_concepts', help='Filter concepts to only include attributes with occurrences in '
                                                 'at least 10 classes', action='store_true')
    args = parser.parse_args()

    train_data, val_data, test_data, mask = extract_data(args.data_dir, args.filter_concepts)

    if not os.path.exists(os.path.join(args.save_dir, 'test.pkl')):

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
    else:
        if os.path.exists(os.path.join(args.save_dir, 'train.pkl')):
            os.rename(os.path.join(args.save_dir, 'train.pkl'), os.path.join(args.save_dir, 'train_0.pkl'))
        if os.path.exists(os.path.join(args.save_dir, 'val.pkl')):
            os.rename(os.path.join(args.save_dir, 'val.pkl'), os.path.join(args.save_dir, 'val_0.pkl'))
        print("Data already present")

    with open(join(args.save_dir, 'mask.pkl'), 'wb') as f:
        pickle.dump(mask, f)

    # generate training and validation sets for the other folds.
    generate_kfold_sets(base_dir=args.save_dir, save_dir=args.save_dir, seed=0)
    