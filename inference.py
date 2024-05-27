"""
Evaluate trained models on the official CUB test set
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import BASE_DIR
from utils import AverageMeter, multiclass_metric, accuracy, binary_accuracy

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def eval_inference(args, use_encoding=False, model_training=True):
    """
    Run inference using model (and model2 if bottleneck)
    length of returned attribute information: (length = N_ATTRIBUTES * N_TEST)
    :param use_encoding: Whether to return model encodings (input of the last bottleneck layer)
    :param model_training: Whether or not the inference is considered as model training, i.e. for data_transformation.
        Only considered for training and validation data.
    Returns: (for notebook analysis)
    all_class_labels: flattened list of class labels for each image
    all_attr_labels: flattened list of labels for each attribute for each image
    all_attr_outputs: flattened list of attribute logits predicted for each attribute for each image
    all_attr_outputs_sigmoid: flattened list of attribute logits predicted (after Sigmoid) for each attribute for each
    image
    """
    # import dataset specific functions and constants
    if args.dataset == 'CUB':
        from CUB.dataset import load_data
        from CUB.config import N_CLASSES, N_ATTRIBUTES
    elif args.dataset == 'CMNIST':
        from CMNIST.dataset import load_data
        from CMNIST.config import N_CLASSES, N_ATTRIBUTES
    elif 'MNIST' in args.dataset or args.dataset == 'SVHN':
        from MNIST.dataset import load_data
        from MNIST.config import N_CLASSES, N_ATTRIBUTES

    # load primary model (bottleneck if x -> y or predictor if c -> y)
    if args.model_dir:
        model = torch.load(args.model_dir)
        model.encodings = use_encoding
        if not hasattr(model, 'use_sigmoid'):
            if args.use_sigmoid:
                model.use_sigmoid = True
            else:
                model.use_sigmoid = False
        model.eval()
        model.to(device)
    else:
        model = None

    # load secondary model (predictor if x -> y)
    if args.model_dir2:
        model2 = torch.load(args.model_dir2)
        if not hasattr(model2, 'use_sigmoid'):
            if args.use_sigmoid:
                model2.use_sigmoid = True
            else:
                model2.use_sigmoid = False
        model2.eval()
        model2.to(device)
    else:
        model2 = None

    attr_acc_meter = AverageMeter()
    class_acc_meter = AverageMeter()

    fold = '_0' if args.fold is None else f'_{args.fold}'
    # create the data loader:
    if not args.eval_data == 'test':
        data_dir = os.path.join(BASE_DIR, args.data_dir, f'{args.eval_data}{fold}.pkl')
        # whether to consider it as model training for the evaluation
        is_model_training = model_training
    else:
        # if there is a data directory for the given fold, use it. Otherwise use the base data
        if os.path.exists(os.path.join(BASE_DIR, args.data_dir, f'{args.eval_data}{fold}.pkl')):
            data_dir = os.path.join(BASE_DIR, args.data_dir, f'{args.eval_data}{fold}.pkl')
        else:
            data_dir = os.path.join(BASE_DIR, args.data_dir, f'{args.eval_data}.pkl')
        is_model_training = False

    # apply noisy transform for the CUB dataset (if set in args)
    if args.dataset == 'CUB':
        loader = load_data([data_dir], is_model_training, args.no_img, args.batch_size, image_dir=args.image_dir,
                           noisy_transform=args.noisy_transform)
    else:
        loader = load_data([data_dir], is_model_training, args.no_img, args.batch_size, image_dir=args.image_dir)

    # prepare outputs
    all_attr_labels, all_attr_outputs, all_attr_outputs_sigmoid = [], [], []
    all_class_labels, all_class_outputs = [], []
    if use_encoding:
        all_encodings = []

    # perform inference
    for data_idx, data in enumerate(loader):
        if args.no_img:  # C -> Y
            inputs, labels = data
            if isinstance(inputs, list):
                inputs = torch.stack(inputs).t().float()
            inputs = inputs.float()
        else:
            inputs, labels, attr_labels = data
            attr_labels = torch.stack(attr_labels).t()  # N x N_ATTRIBUTES

        inputs = inputs.to(device)
        labels = labels.to(device)

        if use_encoding:
            outputs, encodings = model(inputs)
        else:
            outputs = model(inputs)

        if args.no_img:  # C -> Y
            class_outputs = outputs
        else:
            if args.bottleneck:
                attr_outputs_sigmoid = [torch.nn.Sigmoid()(o) for o in outputs]
                if args.use_sigmoid:
                    attr_outputs = attr_outputs_sigmoid
                else:
                    attr_outputs = outputs
                stage2_inputs = torch.cat(attr_outputs, dim=1)
                class_outputs = model2(stage2_inputs)
            else:  # end2end
                attr_outputs_sigmoid = [torch.nn.Sigmoid()(o) for o in outputs[1:]]
                if args.use_sigmoid:
                    attr_outputs = attr_outputs_sigmoid
                else:
                    attr_outputs = outputs[1:]

                class_outputs = outputs[0]

            for i in range(N_ATTRIBUTES):
                acc = binary_accuracy(attr_outputs_sigmoid[i].squeeze(), attr_labels[:, i])
                acc = acc.data.cpu().numpy()
                attr_acc_meter.update(acc, inputs.size(0))

            attr_outputs = torch.cat([o.unsqueeze(1) for o in attr_outputs], dim=1)
            attr_outputs_sigmoid = torch.cat([o for o in attr_outputs_sigmoid], dim=1)
            all_attr_outputs.extend(list(attr_outputs.flatten().data.cpu().numpy()))
            all_attr_outputs_sigmoid.extend(list(attr_outputs_sigmoid.flatten().data.cpu().numpy()))
            all_attr_labels.extend(list(attr_labels.flatten().data.cpu().numpy()))

        all_class_labels.extend(list(labels.data.cpu().numpy()))
        all_class_outputs.extend(np.argmax(class_outputs.data.cpu().numpy(), axis=1))
        if use_encoding:
            all_encodings.extend(list(encodings.data.cpu().numpy()))

        np.set_printoptions(threshold=sys.maxsize)
        class_acc = accuracy(class_outputs, labels, topk=(1, ))[0]  # only class prediction accuracy

        class_acc_meter.update(class_acc, inputs.size(0))

    print(f'Average class accuracy: {(class_acc_meter.avg).item():.5f}')

    all_class_outputs_int = np.array(all_class_outputs) >= 0.5
    _, report = multiclass_metric(all_class_outputs_int, all_class_labels)
    print(report)

    if not args.no_img:  # print some metrics for attribute prediction performance
        print('Average attribute accuracy: %.5f' % attr_acc_meter.avg)
        all_attr_outputs_int = np.array(all_attr_outputs_sigmoid) >= 0.5

        balanced_acc, report = multiclass_metric(all_attr_outputs_int, all_attr_labels)

        # prediction of attributes "classes"
        attr_pre_comb = np.reshape(all_attr_outputs_int, (-1, N_ATTRIBUTES))
        attr_lab_comb = np.reshape(all_attr_labels, (-1, N_ATTRIBUTES))
        digit_preds = np.zeros([N_ATTRIBUTES, 4])
        for d in range(attr_lab_comb.shape[0]):
            l = np.argmax(attr_lab_comb[d])
            if attr_pre_comb[d, l] == 1:
                digit_preds[l, 0] += 1
            else:
                digit_preds[l, 1] += 1
            if all_class_labels[d] == all_class_outputs[d]:
                digit_preds[l, 2] += 1
            else:
                digit_preds[l, 3] += 1

        print("Individual attributes:", "Attr cor, attr_wro, class(attr) cor, class(attr) wro",
              digit_preds, "Total:", np.sum(digit_preds, axis=0), sep='\n')

        f1 = f1_score(all_attr_labels, all_attr_outputs_int)
        print("Total 1's predicted:", sum(np.array(all_attr_outputs_sigmoid) >= 0.5) / len(all_attr_outputs_sigmoid))
        print('Avg attribute balanced acc: %.5f' % (balanced_acc))
        print("Avg attribute F1 score: %.5f" % f1)
        print(report + '\n')
    if use_encoding:
        return all_class_labels, all_attr_labels, all_attr_outputs, all_attr_outputs_sigmoid, all_encodings
    else:
        return all_class_labels, all_attr_labels, all_attr_outputs, all_attr_outputs_sigmoid


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('dataset', type=str, help='Name of the dataset.',
                        choices=['CUB', 'MNIST', 'SVHN', 'MNIST_unbalanced', 'CMNIST'], default='CUB')
    parser.add_argument('-log_dir', default='.', help='where results are stored')
    parser.add_argument('-model_dir', default=None, help='where the trained models are saved')
    parser.add_argument('-model_dir2', default=None, help='where another trained model are saved (for bottleneck only)')
    parser.add_argument('-eval_data', default='test', help='Type of data (train/ val/ test) to be used')
    parser.add_argument('-no_img', help='if included, only use concepts for class prediction', action='store_true')
    parser.add_argument('-bottleneck', help='whether to predict concepts before class labels', action='store_true')
    parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
    parser.add_argument('-data_dir', default='data_CUB/CUB_processed/class_filtered_10',
                        help='directory to the data used for evaluation')
    parser.add_argument('-fold', default=None, help='Evaluation fold. None or 1 to 4 '
                                                    '(None equals to 0 the default split).')
    parser.add_argument('-use_sigmoid', help='Whether to include sigmoid activation before using concepts to predict Y.'
                                             ' For end2end & bottleneck model', action='store_true')
    parser.add_argument('-noisy_transform', choices=['jitter', 'blur', 'erase', 'salt', 'speckle'], default=None,
                        help='Dataset augmentation for CUB')
    args = parser.parse_args()
    args.batch_size = 16

    eval_inference(args)

