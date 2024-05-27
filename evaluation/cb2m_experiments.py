import os
import json
import random
import numpy as np
import torch
import sklearn
from scipy.special import softmax
import rtpt
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cb2m_utils import *
from cb2m.closer_metrics import ectp_precompute
from cb2m.memory_module import MemoryModule

rtpt = rtpt.RTPT(name_initials="DS", experiment_name="CBM interventions", max_iterations=1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def data_precomputation(args):
    """
    Computes info about train, val and test splits, as well as shifted test data.
    Stores the information in pickle files in the directory: precomputed.
    """
    print(f"\n\nPrecompute data for {args.dataset}, fold: {args.fold}")
    if args.dataset == 'CUB':
        noisy = False
        shifted = False
    else:
        noisy = False
        shifted = False
    # load and prepare model2
    model2 = torch.load(args.model_dir2)
    model2.eval()
    model2.to(device)
    fold = f'_{args.fold}'

    if shifted:
        for t_shift in ['fixed', 'random', 'black']:
            args.test_shift = t_shift
            args.split = 'test'
            shift_info = prepare_test_shift(args, model2, N_ATTRIBUTES)
            store_prepared_data(shift_info, f'{t_shift}{fold}', dataset=args.dataset)
    data_splits = ['train', 'val', 'test']
    train_info, val_info, test_info = prepare_data(data_splits, args, model2, N_ATTRIBUTES)

    store_prepared_data(train_info, f'train{fold}', dataset=args.dataset)
    store_prepared_data(val_info, f'val{fold}', dataset=args.dataset)
    store_prepared_data(test_info, f'test{fold}', dataset=args.dataset)

    if noisy:
        data_splits = ['test']
        for n in ['jitter', 'blur', 'erase', 'salt', 'speckle']:
            args.noisy_transform = n
            train_info_n, val_info_n, test_info_n = prepare_data(data_splits, args, model2, N_ATTRIBUTES)
            store_prepared_data(test_info_n, f'test{fold}_aug_{n}', dataset=args.dataset)


def hyperparameter_selection(args):
    """
    Evaluate the selection of instances.
    Iterate over different variants of the parameters: k, select_t, setup_t.
    Reports the total number of selected instances, the class_accuracy of the selected instances before interventions
    as well as the fraction of wrongly classified instances selected.
    Stores the result as a json file.
    """
    # prepare data
    fold = f'_{args.fold}'
    train_info = load_prepared_data(f'train{fold}', args.dataset)
    val_info = load_prepared_data(f'val{fold}', args.dataset)
    attr_group_dict = get_attr_group_dict(args.data_dir, args.dataset)

    # hyperparameter values (for CUB)
    k_val = [1, 2, 3, 4, 5]
    set_t_val = [1.0]
    sel_t_val = range(1, 13)
    results = {}

    # compute class correct and wrong predictions for the whole val set
    comp = val_info['class_outputs'] == val_info['class_labels']
    tot = np.size(comp)
    results[0] = {
        'test_size': tot,
        'test_acc': np.mean(comp),
        'k_values': k_val,
        'setup_t_values': set_t_val,
        'select_t_values': sel_t_val,
    }

    result_prec = np.zeros((len(k_val), len(set_t_val), len(sel_t_val)))
    result_rec = np.zeros(result_prec.shape)

    # iterate over all hyperparameter settings
    for k_idx, k in enumerate(k_val):
        result_k = {}
        for set_idx, setup_t in enumerate(set_t_val):
            result_set = {}
            for sel_idx, select_t in enumerate(sel_t_val):
                # setup memory_module
                memory_module = MemoryModule(k)
                setup(train_info, memory_module, attr_group_dict, setup_t)
                memory_module.prepare_eval()

                # select instances for intervention:
                intervention_ids = select_intervention_instances(val_info, memory_module, threshold=select_t)

                if len(intervention_ids) == 0:
                    # not exactly zero to avoid division through zero for the f1 calculation
                    prec = 0.000000001
                    rec = 0.000000001
                else:
                    prec = 1 - np.mean(comp[intervention_ids])
                    rec = (len(intervention_ids) - np.sum(comp[intervention_ids])) / (tot - np.sum(comp))
                    if prec + rec == 0:
                        prec = 0.000000001
                        rec = 0.000000001

                result_prec[k_idx, set_idx, sel_idx] = prec
                result_rec[k_idx, set_idx, sel_idx] = rec
                result_set[select_t] = {
                    'prec': prec,
                    'rec': rec,
                }

                print(f'k:{k}\tselect_t: {select_t}\tsetup_t: {setup_t}\t '
                      f'#In: {len(intervention_ids)} \t Precision: {prec:.4f}\t Recall: {rec:.4f}\t')
            result_k[setup_t] = result_set
        results[k] = result_k

    result_f1 = 2 * result_prec * result_rec / (result_prec + result_rec)

    # report best hyperparameter settings:
    b_p = np.unravel_index(np.argmax(result_prec, axis=None), result_prec.shape)
    b_r = np.unravel_index(np.argmax(result_rec, axis=None), result_rec.shape)
    b_f = np.unravel_index(np.argmax(result_f1, axis=None), result_f1.shape)

    print(f"\nBest precision settings:\t k:{k_val[b_p[0]]}\tsetup_t:{set_t_val[b_p[1]]}\tselect_t:{sel_t_val[b_p[2]]}")
    print(f"Best recall settings:\t k:{k_val[b_r[0]]}\tsetup_t:{set_t_val[b_r[1]]}\tselect_t:{sel_t_val[b_r[2]]}")
    print(f"Best f1 settings:\t k:{k_val[b_f[0]]}\tsetup_t:{set_t_val[b_f[1]]}\tselect_t:{sel_t_val[b_f[2]]}")

    return k_val[b_f[0]], set_t_val[b_f[1]], sel_t_val[b_f[2]]


def hyperparameter_generalization(args):
    """
    Find a value for the hyperparameter general_t.
    To do that, fill the retriever with interventions on the training set.
    Retrieve and evaluate the retrieved interventions on the validation set.
    """
    # Load data:
    fold = '_0' if args.fold is None else f'_{args.fold}'
    val_info = load_prepared_data(f'train{fold}', args.dataset)
    train_info = load_prepared_data(f'val{fold}', args.dataset)
    attr_group_dict = get_attr_group_dict(args.data_dir, args.dataset)

    # load and prepare model2
    model2 = torch.load(args.model_dir2)
    model2.eval()
    model2.to(device)

    # Detect misclassified examples on the validation set:
    intervention_ids = select_intervention_instances_perfect(val_info)

    # Fill the memory_module with interventions on the validation set
    # get ectp info
    class_out_0_combined, class_out_1_combined = ectp_precompute(val_info, N_ATTRIBUTES, model2)
    ectp_info = [class_out_0_combined, class_out_1_combined]

    memory_module = MemoryModule(k=1)
    fill_memory(intervention_ids, val_info, attr_group_dict, memory_module, ectp_info, use_invisible=False)
    memory_module.prepare_eval()

    # Evaluate the retrieved interventions on the training set.
    # Evaluation is based on class acc when using the retrieved concepts

    general_t_values = np.arange(1.0, 8.0, 0.5)
    res_t = -1
    for general_t in general_t_values:
        n_results = evaluate_generalization(memory_module, train_info, model2, general_t, len(attr_group_dict.keys()))

        print(f"t: {general_t}\tcl_acc_bef: {n_results['cl_acc_bef']:.4f}\tcl_acc_aft: {n_results['cl_acc_aft']:.4f}")
        if n_results['cl_acc_aft'] > n_results['cl_acc_bef']:
            res_t = general_t

    print(f"Optimal t: {res_t}")
    return res_t


def hyperparameter_softmax(args):
    """
        Perform threshold selection for the softmax baseline.
    """
    # prepare data
    fold = f'_{args.fold}'
    val_info = load_prepared_data(f'val{fold}', args.dataset)

    # hyperparameter values
    t_val = np.arange(0.05, 1.00, 0.05)

    # compute class correct and wrong predictions for the whole val set
    comp = val_info['class_outputs'] == val_info['class_labels']
    tot = np.size(comp)

    result_prec = np.zeros(t_val.shape)
    result_rec = np.zeros(result_prec.shape)

    class_softmax = softmax(val_info['class_logits'], axis=1)
    max_val_softmax = np.max(class_softmax, axis=1)

    for t_idx, t in enumerate(t_val):
        # select instances for intervention:
        intervention_ids = np.argwhere(max_val_softmax < t).flatten()

        if len(intervention_ids) == 0:
            # not exactly zero to avoid division through zero for f1 score
            prec = 0.0000001
            rec = 0.0000001
        else:
            prec = 1 - np.mean(comp[intervention_ids])
            rec = (len(intervention_ids) - np.sum(comp[intervention_ids])) / (tot - np.sum(comp))
        result_prec[t_idx] = prec
        result_rec[t_idx] = rec

        print(f't:{t:.2f}\t #In: {len(intervention_ids)} \t Precision: {prec:.4f}\t Recall: {rec:.4f}\t')

    result_f1 = 2 * result_prec * result_rec / (result_prec + result_rec)
    # report best hyperparameter settings:
    b_p = np.argmax(result_prec)
    b_r = np.argmax(result_rec)
    b_f = np.argmax(result_f1)

    print(f"\nBest precision settings:\t t:{t_val[b_p]}")
    print(f"Best recall settings:\t t:{t_val[b_r]}")
    print(f"Best f1 settings:\t t:{t_val[b_f]}")

    return t_val[b_f]


def run_detection_experiment(args):
    """
    Evaluates the performance on the detection task, i.e. detecting wrongly classified examples on the test set
        Evaluates CIR and the baselines (random and softmax). Dataset can be the normal CUB version and the
        shifted versions.
    """

    np.random.seed(args.seed)
    fold = f'_{args.fold}'
    # prepare data
    val_info = load_prepared_data(f'val{fold}', args.dataset)
    if args.test_shift is None:
        test_info = load_prepared_data(f'test{fold}', args.dataset)
    else:
        test_info = load_prepared_data(args.test_shift)
    attr_group_dict = get_attr_group_dict(args.data_dir, args.dataset)

    # if no hyperparameters have been specified, load them from the file.
    with open(os.path.join(args.log_dir, args.dataset, 'hyperparameter.json'), 'r') as fi:
        hyperparameter = json.load(fi)
    k = args.k if args.k is not None else hyperparameter[args.fold]['k']
    setup_t = args.setup_t if args.setup_t is not None else hyperparameter[args.fold]['setup_t']
    select_t = args.select_t if args.select_t is not None else hyperparameter[args.fold]['select_t']
    softmax_t = args.softmax_t if args.softmax_t is not None else hyperparameter[args.fold]['softmax_t']

    # comparison on the test set:
    comp = test_info['class_outputs'] == test_info['class_labels']
    # 1 if correct classified, else 0
    y_true = comp.astype(int)
    tot = np.size(comp)
    img_ids = list(range(len(test_info['class_outputs'])))

    # memory module preparations
    memory_module = MemoryModule(k)
    setup(val_info, memory_module, attr_group_dict, setup_t)
    memory_module.prepare_eval()
    # select instances for interventions:
    intervention_ids = select_intervention_instances(test_info, memory_module, select_t)
    # precision, recall and f1 score
    cb2m_prec = 1 - np.mean(comp[intervention_ids])
    cb2m_rec = (len(intervention_ids) - np.sum(comp[intervention_ids])) / (tot - np.sum(comp))
    cb2m_f1 = 2 * cb2m_prec * cb2m_rec / (cb2m_prec + cb2m_rec)
    # auroc and aupr scores
    cb2m_score = [memory_module.closest_distance(k) for k in test_info['encodings']]
    # normalize the cb2, score to be within [0, 1]
    cb2m_score = (cb2m_score - np.ones(len(cb2m_score)) * min(cb2m_score)) / (max(cb2m_score) - min(cb2m_score))
    cb2m_auroc = sklearn.metrics.roc_auc_score(y_true, cb2m_score)
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, cb2m_score)
    cb2m_aupr = sklearn.metrics.auc(recall, precision)

    # random
    # precision, recall and f1 score
    rand_ids = random.sample(img_ids, len(intervention_ids))
    rand_prec = 1 - np.mean(comp[rand_ids])
    rand_rec = (len(rand_ids) - np.sum(comp[rand_ids])) / (tot - np.sum(comp))
    if rand_prec == 0 and rand_rec == 0:
        rand_f1 = 0.0
    else:
        rand_f1 = 2 * rand_prec * rand_rec / (rand_prec + rand_rec)

    # auroc and aupr score
    random_scores = np.random.rand(len(y_true))
    rand_auroc = sklearn.metrics.roc_auc_score(y_true, random_scores)
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, random_scores)
    rand_aupr = sklearn.metrics.auc(recall, precision)

    # softmax predictions
    class_softmax = softmax(test_info['class_logits'], axis=1)
    max_val_softmax = np.max(class_softmax, axis=1)
    intervention_ids = np.argwhere(max_val_softmax < softmax_t).flatten()
    # precision, recall and f1
    soft_prec = 1 - np.mean(comp[intervention_ids])
    soft_rec = (len(intervention_ids) - np.sum(comp[intervention_ids])) / (tot - np.sum(comp))
    soft_f1 = 2 * soft_prec * soft_rec / (soft_prec + soft_rec)
    # auroc and aupr
    y_score_softmax = np.max(class_softmax, axis=1)
    soft_auroc = sklearn.metrics.roc_auc_score(y_true, y_score_softmax)
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_score_softmax)
    soft_aupr = sklearn.metrics.auc(recall, precision)

    # combined softmax and cb2m:
    combined_score = np.maximum(cb2m_score, y_score_softmax)
    combined_auroc = sklearn.metrics.roc_auc_score(y_true, combined_score)
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, combined_score)
    combined_aupr = sklearn.metrics.auc(recall, precision)

    # Report results
    print(f"Detection on test set (fold: {args.fold}):")
    print(f"Num instances: {len(intervention_ids)}")
    print(f"Random: \tPrecision: {rand_prec:.4f}\tRecall: {rand_rec:.4f}\tF1: {rand_f1:.4f}\t"
          f"AUROC: {rand_auroc:.4f}\tAUPR: {rand_aupr:.4f}")
    print(f"Softmax: \tPrecision: {soft_prec:.4f} \tRecall: {soft_rec:.4f} \tF1: {soft_f1:.4f}\t"
          f"AUROC: {soft_auroc:.4f}\tAUPR: {soft_aupr:.4f}")
    print(f"memory: \t\tPrecision: {cb2m_prec:.4f} \tRecall: {cb2m_rec:.4f} \tF1: {cb2m_f1:.4f}\t"
          f"AUROC: {cb2m_auroc:.4f}\tAUPR: {cb2m_aupr:.4f}")
    print(f"Combined: \t\tPrecision: {0:.4f} \tRecall: {0:.4f} \tF1: {0:.4f}\t"
          f"AUROC: {combined_auroc:.4f}\tAUPR: {combined_aupr:.4f}")

    return (rand_auroc, rand_aupr), (soft_auroc, soft_aupr), (cb2m_auroc, cb2m_aupr), (combined_auroc, combined_aupr)


def run_performance_experiment(args):
    """
    Evaluate the accuracy of the full model after performing interventions.
    All Methods use the same instances selected for intervention, determined by the parameters (k, setup_t, select_t).
    For a single method or all specified methods, interventions are done on one to all concept groups
    Additional parameter: args.baseline can be either random or softmax, if empty, the memory module is used
    """
    fold = f'_{args.fold}'
    # load and prepare model2
    model2 = torch.load(args.model_dir2)
    model2.eval()
    model2.to(device)

    # prepare data
    if args.test_shift is not None:
        if args.test_shift not in ['fixed', 'random', 'black']:
            raise ValueError(f"Unknown version of test shift specified: {args.test_shift}.")
        test_info = load_prepared_data(args.test_shift)
    else:
        test_info = load_prepared_data(f'test{fold}', args.dataset)

    val_info = load_prepared_data(f'val{fold}', args.dataset)
    attr_group_dict = get_attr_group_dict(args.data_dir, args.dataset)
    n_groups = len(attr_group_dict.keys())

    # if no hyperparameters have been specified, load them from the file.
    with open(os.path.join(args.log_dir, args.dataset, 'hyperparameter.json'), 'r') as fi:
        hyperparameter = json.load(fi)
    k = args.k if args.k is not None else hyperparameter[args.fold]['k']
    setup_t = args.setup_t if args.setup_t is not None else hyperparameter[args.fold]['setup_t']
    select_t = args.select_t if args.select_t is not None else hyperparameter[args.fold]['select_t']
    softmax_t = args.softmax_t if args.softmax_t is not None else hyperparameter[args.fold]['softmax_t']

    # setup memory_module
    memory_module = MemoryModule(k)
    setup(val_info, memory_module, attr_group_dict, setup_t)
    memory_module.prepare_eval()

    # select instances for intervention:
    intervention_ids = select_intervention_instances(test_info, memory_module, threshold=select_t)
    if args.baseline == 'random':
        intervention_ids = random.sample(range(len(test_info['class_labels'])), len(intervention_ids))
    elif args.baseline == 'softmax':
        class_softmax = softmax(test_info['class_logits'], axis=1)
        max_val_softmax = np.max(class_softmax, axis=1)
        intervention_ids = np.argwhere(max_val_softmax < softmax_t).flatten()
    elif args.baseline == 'combined':
        class_softmax = softmax(test_info['class_logits'], axis=1)
        max_val_softmax = np.max(class_softmax, axis=1)
        soft_ids = np.argwhere(max_val_softmax < softmax_t).flatten()
        intervention_ids = list(set(intervention_ids).union(soft_ids))

    results = {}
    # compute class correct and wrong predictions for the whole test set and only selected instances for interventions
    compare_cl_before = test_info['class_outputs'] == test_info['class_labels']
    cl_acc_before = np.mean(compare_cl_before)
    test_cor = np.sum(compare_cl_before)
    test_wro = compare_cl_before.shape[0] - test_cor
    compare_cl_selected = compare_cl_before[intervention_ids]
    cl_acc_before_sel = np.mean(compare_cl_before[intervention_ids])
    select_cor = np.sum(compare_cl_selected)
    select_wro = compare_cl_selected.shape[0] - select_cor

    # intervention ids for attributes to compute concept accuracy statistics
    attr_intervention_ids = []
    for idx in intervention_ids:
        attr_intervention_ids.extend(list(range(N_ATTRIBUTES * idx, N_ATTRIBUTES * (idx + 1))))
    # compare concept accuracy statistics
    compare_attr_before = test_info['attr_binary_outputs'] == test_info['attr_labels']
    test_attr_acc = np.mean(compare_attr_before)
    sel_attr_acc = np.mean(compare_attr_before[attr_intervention_ids])
    oth_attr_acc = np.mean(np.delete(compare_attr_before, attr_intervention_ids))
    results[0] = {
        'test_cor': int(test_cor),
        'test_wro': int(test_wro),
        'select_cor': int(select_cor),
        'select_wro': int(select_wro),
        'test_attr_acc': test_attr_acc,
        'other_attr_acc': oth_attr_acc,
    }

    # print initial information to the console
    cl_acc_test = test_cor / (compare_cl_before.shape[0])
    cl_acc_select = select_cor / (compare_cl_selected.shape[0])
    print(f"Size of test set: \t{compare_cl_before.shape[0]}. "
          f"\tClass accuracy on the test set: {cl_acc_test: .4f}")
    print(f"Number of selections: \t{compare_cl_selected.shape[0]}. "
          f"\tClass accuracy on the selected instances before interventions: {cl_acc_select: .4f}")
    print(f"Fraction of wrongly classified instances selected: {select_wro / test_wro: .4f}")

    # pre computations for ectp
    if args.method == 'ectp' or args.method == 'all':
        # compute the changed predictions if interventions would occur:
        class_out_0_combined, class_out_1_combined = ectp_precompute(test_info, N_ATTRIBUTES, model2)
        ectp_info = [class_out_0_combined, class_out_1_combined]
    else:
        ectp_info = None

    if args.method == 'all':
        methods = ['rand', 'ucp', 'ectp', 'lcp']
    else:   # single method specified
        methods = [args.method]

    for n_concepts in [0, 10, n_groups]: #range(1, n_groups + 1):
        n_results = {}
        for method in methods:
            global_interventions = select_interventions(intervention_ids, test_info, attr_group_dict, method,
                                                        n_concepts, ectp_info)

            n_results[method] = evaluate_interventions(test_info, intervention_ids, global_interventions, model2,
                                                       args.use_invisible)

        results[n_concepts] = n_results

        # print information about the current step
        attr_acc_bef = f'Attr acc before interventions: \t'
        class_acc_aft = f'Class acc after interventions: \t'
        for method in methods:
            attr_acc_bef += f'{method}: {results[n_concepts][method]["attr_acc_bef"]:.4f}\t'
            class_acc_aft += f'{method}: {results[n_concepts][method]["cl_acc_aft"]:.4f}\t'
        print(f'N Concept: {n_concepts}')
        print(attr_acc_bef)
        print(class_acc_aft)

    # log the results
    results[-1] = {'method': args.method,
                   'setup_t': args.setup_t,
                   'select_t': args.select_t,
                   'k': args.k,
                   'baseline': args.baseline}

    if args.dataset == 'CUB':
        return (sel_attr_acc, oth_attr_acc, test_attr_acc), (cl_acc_before, cl_acc_before_sel), \
            (results[10]['ectp']['cl_acc_aft'], results[n_groups]['ectp']['cl_acc_aft'],
             results[10]['ectp']['cl_acc_aft_tot'], results[n_groups]['ectp']['cl_acc_aft_tot'])
    else:
        return (sel_attr_acc, oth_attr_acc, test_attr_acc), (cl_acc_before, cl_acc_before_sel), \
            (None, results[n_groups]['ectp']['cl_acc_aft'], None, results[n_groups]['ectp']['cl_acc_aft_tot'])


def generalization_experiment(args):
    """
    Perform generalization experiments:
    Performance improvement of the model when reusing interventions.

    """
    # Parameters: Setup_t, Select_t, k, n_concepts, model_dir2, general_t
    # Load data:
    fold = f'_{args.fold}'
    if args.dataset == 'CUB':
        train_info = load_prepared_data(f'val{fold}', args.dataset)
        val_info = load_prepared_data(f'test{fold}', args.dataset)
        test_info = load_prepared_data(f'test{fold}_aug_' + args.test_aug, args.dataset)
    else:
        train_info = load_prepared_data(f'train{fold}', args.dataset)
        val_info = load_prepared_data(f'val{fold}', args.dataset)
        test_info = load_prepared_data(f'test{fold}', args.dataset)
    attr_group_dict = get_attr_group_dict(args.data_dir, args.dataset)
    n_groups = len(attr_group_dict.keys())

    # if no hyperparameters have been specified, load them from the file.
    with open(os.path.join(args.log_dir, args.dataset, 'hyperparameter.json'), 'r') as fi:
        hyperparameter = json.load(fi)
    k = args.k if args.k is not None else hyperparameter[str(args.fold)]['k']
    setup_t = args.setup_t if args.setup_t is not None else hyperparameter[str(args.fold)]['setup_t']
    select_t = args.select_t if args.select_t is not None else hyperparameter[str(args.fold)]['select_t']
    general_t = args.general_t if args.general_t is not None else hyperparameter[str(args.fold)]['general_t']

    # load and prepare model2
    model2 = torch.load(args.model_dir2)
    model2.eval()
    model2.to(device)

    intervention_ids = select_intervention_instances_perfect(val_info)
    if args.data_frac < 1.0:
        k_elems = int(len(intervention_ids) * args.data_frac)
        intervention_ids = random.sample(intervention_ids, k_elems)

    # Fill CIR with the interventions on the selected test_examples
    # get ectp info
    class_out_0_combined, class_out_1_combined = ectp_precompute(val_info, N_ATTRIBUTES, model2)
    ectp_info = [class_out_0_combined, class_out_1_combined]

    memory_module = MemoryModule(k=1)
    fill_memory(intervention_ids, val_info, attr_group_dict, memory_module, ectp_info, use_invisible=False)
    memory_module.prepare_eval()

    results = {}
    # Compute base statistics for the noisy test set, record them
    compare_cl_before = test_info['class_outputs'] == test_info['class_labels']
    compare_attr_before = test_info['attr_labels'] == test_info['attr_binary_outputs']
    cl_acc_test = np.mean(compare_cl_before)
    attr_acc_test = np.mean(compare_attr_before)
    results[0] = {
        'test_cl_acc': float(cl_acc_test),
        'test_attr_acc': float(attr_acc_test)
    }

    # Evaluate CIR on the noisy test set:
    for n_concepts in range(1, n_groups + 1):
        n_results = evaluate_generalization(memory_module, test_info, model2, general_t, n_concepts)
        results[n_concepts] = n_results

        if n_concepts == 1:
            print(f"Size of test set: \t{compare_cl_before.shape[0]}. "
                  f"\tClass accuracy on the test set: {cl_acc_test:.4f} "
                  f"\tAttribute accuracy on the test set: {attr_acc_test:.4f}")
            print(f"Number of selections: \t{n_results['num_int']}. "
                  f"\tClass accuracy on the selected instances before interventions: {n_results['cl_acc_bef']:.4f}")

        # print information about the current step
        print(f"N Concept: {n_concepts}")
        print(f"Attribute Accuracies (intervened attributes): \t Before interventions: {n_results['attr_acc_bef']:.4f}"
              f"\tAfter interventions: {n_results['attr_acc_aft']:.4f}")
        print(f"Class Accuracies: Intervened examples after intervention: {n_results['cl_acc_aft']:.4f}\t"
              f"Full test set after interventions: {n_results['cl_acc_aft_tot']:.4f}")

    # Record results
    # log the results
    results[-1] = {'setup_t': setup_t,
                   'select_t': select_t,
                   'k': k,
                   'general_t': general_t,
                   'test_aug': args.test_aug}

    if args.data_frac != 1.0:
        file_name = f'generalization{fold}_{args.test_aug}_{args.data_frac*100}.json'
    else:
        file_name = f'generalization{fold}_{args.test_aug}.json'
    os.makedirs(os.path.join(args.log_dir, args.dataset), exist_ok=True)
    with open(os.path.join(args.log_dir, args.dataset, file_name), 'w') as fi:
        json.dump(results, fi)


def parse_arguments():
    parser = argparse.ArgumentParser(description='TTI Estimations')
    # experiment type:
    parser.add_argument('exp', choices=['precompute', 'hyperparameter', 'performance', 'detection', 'generalization'],
                        help='Experiment name')
    parser.add_argument('dataset', type=str, help='Name of the dataset.',
                        choices=['CUB', 'MNIST', 'MNIST_unbalanced', 'SVHN', 'CMNIST',
                                 'MNIST_nozero', 'MNIST_noone', 'MNIST_notwo', 'MNIST_nothree', 'MNIST_nofour',
                                 'MNIST_nofive', 'MNIST_nosix', 'MNIST_noseven', 'MNIST_noeight'])
    parser.add_argument('-seed', default=0)

    # evaluation arguments
    parser.add_argument('-log_dir', default='.', help='where results are stored')
    parser.add_argument('-model_dir', default=None, help='path to XtoC model')
    parser.add_argument('-model_dir2', default=None, help='path to the CtoY model')
    parser.add_argument('-image_dir', default='images', help='image folder to run inference on')
    parser.add_argument('-data_dir', default='data_CUB/CUB_processed/class_filtered_10',
                        help='data directory containing the pickle files')
    parser.add_argument('-no_img', help='if included, only use attributes (and not raw imgs) for class prediction',
                        action='store_true')
    parser.add_argument('-bottleneck', help='whether to predict attributes before class labels', action='store_true')
    parser.add_argument('-use_sigmoid', help='Whether to include sigmoid activation before attributes.',
                        action='store_true')
    parser.add_argument('-batch_size', default=16, help='batch size for training')
    parser.add_argument('-noisy_transform', default=False, action='store_true',
                        help='Apply noise transformations to the data')

    # intervention evaluation arguments
    parser.add_argument('-method', default='ectp', help='Method to select concepts. Of: ucp, lcp, ectp, rand, all')
    parser.add_argument('-use_invisible', action='store_true',
                        help='Whether interventions from invisible concepts are allowed')
    parser.add_argument('-setup_t', default=None, type=float, help='Accuracy threshold to setup the memory module.')
    parser.add_argument('-select_t', default=None, type=float, help='Distance threshold for the memory module.')
    parser.add_argument('-k', default=None, type=int, help='Number of neighbors for the memory module.')
    parser.add_argument('-general_t', default=None, type=float, help='Distance threshold for CIR generalization.')
    parser.add_argument('-softmax_t', default=None, type=float, help='threshold for the softmax baseline')
    parser.add_argument('-test_aug',
                        help='Augmented test version for data generalization: jitter, blur, erase, salt, speckle.')
    parser.add_argument('-baseline', help='Evaluate a baseline instead of the memory module: random or softmax')
    parser.add_argument('-test_shift', help='Evaluate the results under a shift in test distribution. Can be one of:'
                                            'fixed, random, black.')
    parser.add_argument('-fold', default=0, help='Evaluation fold (for RQ1, RQ2, RQ4). 0 to 4.')

    parser.add_argument('-data_frac', type=float, default=1.0, help="Fraction of the val data which CB2M"
                                                                    "can see for the generalization experiments.")

    args = parser.parse_args()

    return args


def run_experiments(args):
    """
    Runs experiments for CB2M according to args.
    Structure of the results dictionary:
    res[1], ... res[n]: Information about interventions on that number of concept (groups)
    res[-1]: General metadata about the experiment: method, setup_t, select_t, k
    res[0]: General information about the selection:
        test_cor: Number of correctly classified instances in the test set
        test_wro: Number of wrongly classified instances in the test set
        select_cor: Number of correctly classified instances among the selected instances for interventions
        select_wro: Number of wrongly classified instances among the selected instances for interventions
        The last two are before any interventions are done.
    """
    rtpt.start()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.exp == 'precompute':
        data_precomputation(args)
    elif args.exp == 'hyperparameter':
        if args.fold in ['0', '1', '2', '3', '4']:
            hyperparameter_softmax(args)
            hyperparameter_selection(args)
            pass
        else:
            results = {}
            for f in [0, 1, 2, 3, 4]:
                args.fold = f
                if args.dataset == 'SVHN':
                    dataset = 'MNIST'
                else:
                    dataset = args.dataset
                args.model_dir2 = f'results/{dataset}/IndependentModel_fold_{f}/best_model.pth'
                opt_soft = hyperparameter_softmax(args)
                opt_k, opt_t_set, opt_t_sel = hyperparameter_selection(args)
                opt_general_t = hyperparameter_generalization(args)
                f_results = {
                    'softmax_t': opt_soft,
                    'setup_t': opt_t_set,
                    'select_t': opt_t_sel,
                    'k': opt_k,
                    'general_t': opt_general_t
                }
                results[f] = f_results
            os.makedirs(os.path.join(args.log_dir, args.dataset), exist_ok=True)
            with open(os.path.join(args.log_dir, args.dataset, 'hyperparameter.json'), 'w') as f:
                json.dump(results, f)

    elif args.exp == 'detection':
        if args.fold in ['0', '1', '2', '3', '4']:
            run_detection_experiment(args)
        else:
            results = {}
            for f in [0, 1, 2, 3, 4]:
                args.fold = str(f)
                if args.dataset == 'SVHN':
                    dataset = 'MNIST'
                else:
                    dataset = args.dataset
                args.model_dir2 = f'results/{dataset}/IndependentModel_fold_{f}/best_model.pth'
                r_rand, r_soft, r_cb2m, r_comb = run_detection_experiment(args)
                f_results = {
                    'rand_auroc': r_rand[0],
                    'rand_aupr': r_rand[1],
                    'soft_auroc': r_soft[0],
                    'soft_aupr': r_soft[1],
                    'cb2m_auroc': r_cb2m[0],
                    'cb2m_aupr': r_cb2m[1],
                    'combined_auroc': r_comb[0],
                    'combined_aupr': r_comb[1],
                }
                results[f] = f_results
            shift = '' if args.test_shift is None else f'_{args.test_shift}'
            os.makedirs(os.path.join(args.log_dir, args.dataset), exist_ok=True)
            with open(os.path.join(args.log_dir, args.dataset, f'detection{shift}_results.json'), 'w') as f:
                json.dump(results, f)

    elif args.exp == 'performance':
        if args.fold in ['0', '1', '2', '3', '4']:
            run_performance_experiment(args)
        else:
            results = {}
            for f in [0, 1, 2, 3, 4]:
                args.fold = str(f)
                if args.dataset == 'SVHN':
                    dataset = 'MNIST'
                else:
                    dataset = args.dataset
                args.model_dir2 = f'results/{dataset}/IndependentModel_fold_{f}/best_model.pth'
                concepts_r, class_r_bef, class_r_aft = run_performance_experiment(args)
                f_results = {
                    'con_acc_sel': concepts_r[0],
                    'con_acc_oth': concepts_r[1],
                    'con_acc_tot': concepts_r[2],
                    'cl_bef_tot': class_r_bef[0],
                    'cl_bef_sel': class_r_bef[1],
                    'cl28': class_r_aft[1],
                    'cl28_tot': class_r_aft[3],
                }
                if args.dataset == 'CUB':
                    f_results['cl10'] = class_r_aft[0]
                    f_results['cl10_tot'] = class_r_aft[2]
                results[f] = f_results
            base = '_' + (args.baseline if args.baseline is not None else 'cb2m')
            os.makedirs(os.path.join(args.log_dir, args.dataset), exist_ok=True)
            with open(os.path.join(args.log_dir, args.dataset, f'performance_results{base}.json'), 'w') as f:
                json.dump(results, f)
    else:
        generalization_experiment(args)


if __name__ == '__main__':
    args = parse_arguments()
    if args.dataset == 'CUB':
        from CUB.config import N_ATTRIBUTES
    elif args.dataset == 'CMNIST':
        from CMNIST.config import N_ATTRIBUTES
    elif 'MNIST' in args.dataset or args.dataset == 'SVHN':
        from MNIST.config import N_ATTRIBUTES
    run_experiments(args)
