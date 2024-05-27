"""
Train InceptionV3 Network using the CUB-200-2011 dataset
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
from torch.nn import functional as F

from utils import Logger, AverageMeter, accuracy, binary_accuracy, find_attribute_imbalance, find_class_imbalance
from config import BASE_DIR, MIN_LR, LR_DECAY_SIZE
from models import bottleneck_model, joint_model, independent_model, mnist_bottleneck, cmnist_bottleneck
from sklearn.metrics import roc_auc_score

import rtpt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_c_to_y_epoch(model, optimizer, loader, loss_meter, acc_meter, criterion, is_training):
    """
    C -> Y: Predicting class labels from concept information
    """
    if is_training:
        model.train()
    else:
        model.eval()
    for _, data in enumerate(loader):
        inputs, labels = data
        labels = labels.to(device)
        if isinstance(inputs, list):
            inputs = torch.stack(inputs).t()
        inputs = torch.flatten(inputs, start_dim=1).float().to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels, topk=(1,))
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc[0], inputs.size(0))

        if is_training:
            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()
            optimizer.step()  # optimizer step to update parameters
    return loss_meter, acc_meter


def run_epoch(model, optimizer, loader, loss_meter, acc_meter, criterion, attr_criterion, args, n_attributes,
              is_training):
    """
    For the rest of the networks (X -> C or X -> C -> Y)
    """
    if is_training:
        model.train()
    else:
        model.eval()
    for _, data in enumerate(loader):
        # extract inputs, labels and attr_labels
        inputs, labels, attr_labels = data
        attr_labels = [i.long() for i in attr_labels]
        attr_labels = torch.stack(attr_labels).t()
        attr_labels_var = torch.autograd.Variable(attr_labels).float().to(device)
        inputs_var = torch.autograd.Variable(inputs).to(device)
        labels_var = torch.autograd.Variable(labels).to(device)

        # training loss with auxiliary logits
        if is_training and args.use_aux:
            outputs, aux_outputs = model(inputs_var)
            losses = []
            out_start = 0
            if not args.bottleneck:  # loss main is for the main task label (always the first output)
                loss_main = 1.0 * criterion(outputs[0], labels_var) + 0.4 * criterion(aux_outputs[0], labels_var)
                losses.append(loss_main)
                out_start = 1
            if args.attr_loss_weight > 0:  # X -> C, end2end
                for i in range(len(attr_criterion)):
                    losses.append(args.attr_loss_weight * (
                                1.0 * attr_criterion[i](outputs[i + out_start].squeeze(), attr_labels_var[:, i])
                                + 0.4 * attr_criterion[i](aux_outputs[i + out_start].squeeze(), attr_labels_var[:, i])))
        else:  # testing or no aux logits
            outputs = model(inputs_var)
            losses = []
            out_start = 0
            if not args.bottleneck:
                loss_main = criterion(outputs[0], labels_var)
                losses.append(loss_main)
                out_start = 1
            if args.attr_loss_weight > 0:  # X -> C, end2end
                for i in range(len(attr_criterion)):
                    losses.append(attr_criterion[i](
                        outputs[i + out_start].squeeze(), attr_labels_var[:, i]))
        if args.bottleneck:  # attribute accuracy
            sigmoid_outputs = torch.sigmoid(torch.cat(outputs, dim=1))
            acc = binary_accuracy(sigmoid_outputs, attr_labels)
            acc_meter.update(acc.data.cpu().numpy(), inputs.size(0))
        else:
            acc = accuracy(outputs[0], labels, topk=(1,))  # only care about class prediction accuracy
            acc_meter.update(acc[0], inputs.size(0))

        if args.bottleneck:
            total_loss = sum(losses) / n_attributes
        else:  # co training, loss by class prediction and loss by attribute prediction have the same weight
            total_loss = losses[0] + sum(losses[1:])
            if args.normalize_loss:
                total_loss = total_loss / (1 + args.attr_loss_weight * n_attributes)

        loss_meter.update(total_loss.item(), inputs.size(0))
        if is_training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    return loss_meter, acc_meter


def train(model, args):
    rtpt_measure = rtpt.RTPT(name_initials="DS", experiment_name="CBM Training", max_iterations=args.epochs)

    if args.dataset == 'CUB':
        from CUB.dataset import load_data
        from CUB.config import N_CLASSES, N_ATTRIBUTES
    elif args.dataset == 'MNIST':
        from MNIST.dataset import load_data
        from MNIST.config import N_CLASSES, N_ATTRIBUTES
    elif args.dataset == 'CMNIST':
        from CMNIST.dataset import load_data
        from CMNIST.config import N_CLASSES, N_ATTRIBUTES

    torch.set_num_threads(50)
    rtpt_measure.start()
    fold = '' if args.fold is None else f'_{args.fold}'
    if args.train_file is not None:
        train_data_path = os.path.join(BASE_DIR, args.data_dir, f'{args.train_file}{fold}.pkl')
    else:
        train_data_path = os.path.join(BASE_DIR, args.data_dir, f'train{fold}.pkl')
    if args.val_file is not None:
        val_data_path = os.path.join(BASE_DIR, args.data_dir, f'{args.val_file}{fold}.pkl')
    else:
        val_data_path = os.path.join(BASE_DIR, args.data_dir, f'val{fold}.pkl')

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = Logger(os.path.join(args.log_dir, 'log.txt'))
    logger.write(str(args) + '\n')
    logger.write('train data path: %s\n' % train_data_path)

    model = model.to(device)
    # handle class imbalance and create class loss
    class_imbalance = find_class_imbalance(train_data_path, N_CLASSES)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_imbalance, device=device, dtype=torch.float))
    # concept loss and concept imbalance
    if args.no_img:
        attr_criterion = None
    else:
        attr_imbalance = find_attribute_imbalance(train_data_path)
        logger.write(str(attr_imbalance) + '\n')
        attr_criterion = []  # separate criterion (loss function) for each attribute
        for ratio in attr_imbalance:
            attr_criterion.append(lambda src, target: F.binary_cross_entropy_with_logits(
                src, target=target, weight=torch.tensor([ratio], device=device, dtype=torch.float)))

    logger.flush()

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                                        weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    if args.ckpt:  # retraining
        train_loader = load_data([train_data_path, val_data_path], True, args.no_img, args.batch_size,
                                 image_dir=args.image_dir, confounded=args.confounded)
        val_loader = None
    else:
        train_loader = load_data([train_data_path], True, args.no_img, args.batch_size, image_dir=args.image_dir, confounded=args.confounded)
        val_loader = load_data([val_data_path], False, args.no_img, args.batch_size, image_dir=args.image_dir, confounded=args.confounded)

    best_val_epoch = -1
    best_val_acc = 0

    for epoch in range(0, args.epochs):
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()

        if args.no_img:
            run_c_to_y_epoch(model, optimizer, train_loader, train_loss_meter, train_acc_meter, criterion,
                             is_training=True)
        else:
            run_epoch(model, optimizer, train_loader, train_loss_meter, train_acc_meter, criterion, attr_criterion,
                      args, N_ATTRIBUTES, is_training=True)

        if args.ckpt:  # retraining on train and val set
            val_loss_meter = train_loss_meter
            val_acc_meter = train_acc_meter
        else:  # evaluate on val set
            val_loss_meter = AverageMeter()
            val_acc_meter = AverageMeter()
            with torch.no_grad():
                if args.no_img:
                    run_c_to_y_epoch(model, optimizer, val_loader, val_loss_meter, val_acc_meter, criterion,
                                     is_training=False)
                else:
                    run_epoch(model, optimizer, val_loader, val_loss_meter, val_acc_meter, criterion, attr_criterion,
                              args, N_ATTRIBUTES, is_training=False)

        if best_val_acc < val_acc_meter.avg:
            best_val_epoch = epoch
            best_val_acc = val_acc_meter.avg
            logger.write('New model best model at epoch %d\n' % epoch)
            torch.save(model, os.path.join(args.log_dir, 'best_model.pth'))
            # in the case of retraining, stop when the model reaches 100% accuracy on both train + val sets
            if best_val_acc >= 100 and args.ckpt:
                break

        train_loss_avg = train_loss_meter.avg
        val_loss_avg = val_loss_meter.avg

        logger.write('Epoch [%d]:\tTrain loss: %.4f\tTrain accuracy: %.4f\t'
                     'Val loss: %.4f\tVal acc: %.4f\t'
                     'Best val epoch: %d\n'
                     % (epoch, train_loss_avg, train_acc_meter.avg, val_loss_avg, val_acc_meter.avg, best_val_epoch))
        logger.flush()

        scheduler.step()  # scheduler step to update lr at the end of epoch
        rtpt_measure.step(f'epoch:{epoch}')
        # inspect lr
        if epoch % 10 == 0:
            print('Current lr:', scheduler.get_last_lr())

        if epoch >= 100 and val_acc_meter.avg < 3:
            print("Early stopping because of low accuracy")
            break
        if epoch - best_val_epoch >= 100:
            print("Early stopping because acc hasn't improved for a long time")
            break


def train_bottleneck(args):
    if args.dataset == 'CUB':
        from CUB.config import N_CLASSES, N_ATTRIBUTES
        model = bottleneck_model(pretrained=args.pretrained, num_classes=N_CLASSES, use_aux=args.use_aux,
                                 n_attributes=N_ATTRIBUTES)
    elif args.dataset == 'MNIST':
        from MNIST.config import N_CLASSES, N_ATTRIBUTES
        model = mnist_bottleneck(n_attributes=N_ATTRIBUTES)
    elif args.dataset == 'CMNIST':
        from CMNIST.config import N_CLASSES, N_ATTRIBUTES
        model = cmnist_bottleneck(n_attributes=N_ATTRIBUTES)
    train(model, args)


def train_independent(args):
    if args.dataset == 'CUB':
        from CUB.config import N_CLASSES, N_ATTRIBUTES
        model = independent_model(n_attributes=N_ATTRIBUTES, num_classes=N_CLASSES)
    elif args.dataset == 'MNIST':
        from MNIST.config import N_CLASSES, N_ATTRIBUTES
        model = independent_model(n_attributes=N_ATTRIBUTES, num_classes=N_CLASSES)
    elif args.dataset == 'CMNIST':
        from CMNIST.config import N_CLASSES, N_ATTRIBUTES
        model = independent_model(n_attributes=N_ATTRIBUTES, num_classes=N_CLASSES)
    train(model, args)


def train_joint(args):
    if args.dataset == 'CUB':
        from CUB.config import N_CLASSES, N_ATTRIBUTES
    elif args.dataset == 'MNIST':
        from MNIST.config import N_CLASSES, N_ATTRIBUTES
    elif args.dataset == 'CMNIST':
        from CMNIST.config import N_CLASSES, N_ATTRIBUTES
    model = joint_model(pretrained=args.pretrained, num_classes=N_CLASSES, use_aux=args.use_aux,
                        n_attributes=N_ATTRIBUTES, use_sigmoid=args.use_sigmoid)
    train(model, args)

def finetune_bottleneck(args):
    assert args.model_dir is not None, "Finetuning requires a model directory to load the model from."
    model = torch.load(args.model_dir)

    train(model, args)


def parse_arguments():
    # Get argparse configs from user
    parser = argparse.ArgumentParser(description='CUB Training')
    parser.add_argument('dataset', type=str, help='Name of the dataset.', choices=['CUB', 'MNIST', 'CMNIST'])
    parser.add_argument('exp', type=str, choices=['Bottleneck', 'Independent', 'Joint', 'Finetune'],
                        help='Name of experiment to run.')

    parser.add_argument('-log_dir', default=None, help='where the trained model is saved')
    parser.add_argument('-batch_size', '-b', type=int, help='mini-batch size')
    parser.add_argument('-epochs', '-e', type=int, help='epochs for training process')
    parser.add_argument('-lr', type=float, help="learning rate")
    parser.add_argument('-weight_decay', type=float, default=5e-5, help='weight decay for optimizer')
    parser.add_argument('-pretrained', '-p', action='store_true',
                        help='whether to load pretrained model & just fine-tune')
    parser.add_argument('-use_aux', action='store_true', help='whether to use aux logits')
    parser.add_argument('-attr_loss_weight', default=1.0, type=float,
                        help='weight for loss by predicting attributes')
    parser.add_argument('-no_img', action='store_true',
                        help='if included, only use attributes (and not raw imgs) for class prediction')
    parser.add_argument('-bottleneck', help='whether to predict attributes before class labels',
                        action='store_true')
    parser.add_argument('-data_dir', default='data_CUB/CUB_processed/class_filtered_10',
                        help='directory to the training data')
    parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
    parser.add_argument('-end2end', action='store_true',
                        help='Whether to train X -> C -> Y end to end.')
    parser.add_argument('-optimizer', default='SGD',
                        help='Type of optimizer to use, options incl SGD, RMSProp, Adam')
    parser.add_argument('-ckpt', action='store_true', help='For retraining on both train + val set')
    parser.add_argument('-normalize_loss', action='store_true',
                        help='Whether to normalize loss by taking attr_loss_weight into account')
    parser.add_argument('-use_sigmoid', action='store_true',
                        help='Whether to include sigmoid activation before using attributes to predict Y. '
                             'For end2end & bottleneck model')
    parser.add_argument('-fold', default=None, help='Evaluation fold (for RQ1, RQ2, RQ4). None or 0 to 4 '
                                                    '(None is the same as 0, the default split).')
    
    parser.add_argument('-model_dir', default=None, help='Enables finetuning the model from the given directory.')
    parser.add_argument('-train_file', default=None, help='Training data (if not standard data setting)')
    parser.add_argument('-val_file', default=None, help='Validation data (if not standard data setting)')
    parser.add_argument('-data_frac', default=1.0, help='Finetuning on a fraction of the data.', type=float)
    parser.add_argument('-confounded', action='store_true', help='if set, uses the confounded CUB images.')


    args = parser.parse_args()

    return args
