import os
import pickle
import numpy as np
import torch

from inference import eval_inference
from utils import get_attribute_groupings
from cb2m.closer_metrics import rand, ucp, lcp, ectp, ectp_precompute

from sklearn.metrics import confusion_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_attr_group_dict(data_dir, dataset):
    """
    Creates an updated attr_group_dict considering the masked attributes.
    """
    # group the attributes based on their connections - e.g. group beak-color::black, beak-color::brown, ...
    mask = pickle.load(open(os.path.join(data_dir, 'mask.pkl'), 'rb'))
    # load the original groupings
    attr_group_dict = get_attribute_groupings(dataset)

    # apply the mask to the mapping: only keep the attributes which are not filtered out
    for group_id, attr_ids in attr_group_dict.items():
        new_attr_ids = []
        for attr_id in attr_ids:
            if attr_id in mask:
                new_attr_ids.append(attr_id)
        attr_group_dict[group_id] = new_attr_ids

    # update the enumeration of the attributes in the group mapping to match the filtered data
    total_so_far = 0
    for group_id, attr_ids in attr_group_dict.items():
        class_attr_ids = list(range(total_so_far, total_so_far + len(attr_ids)))
        total_so_far += len(attr_ids)
        attr_group_dict[group_id] = class_attr_ids

    return attr_group_dict


def prepare_data(data_splits, args, model2, n_attributes):
    """
    Load the data (train/test/val splits as required).
    Get the attribute-group mapping (and update it based on the mask).
    Perform inference once on all specified splits.
    Parameters:
        data_splits: list of any combination of 'train', 'val', 'test'. At least one is required
        args: further kwargs (data_dir, args for eval)
        model2: Second stage model to convert attribute predictions into class predictions
    returns:
        [train_info, val_info, test_info]
        where, train, val and test_info are dictionaries containing the inference results for the respective sets
        or empty dictionaries if the split is not specified within data_splits
    """

    # load the attribute mask
    mask = pickle.load(open(os.path.join(args.data_dir, 'mask.pkl'), 'rb'))
    results = []
    fold = '' if args.fold is None else f'_{args.fold}'

    # load the data
    if 'train' in data_splits:
        train_data = pickle.load(open(os.path.join(args.data_dir, f'train{fold}.pkl'), 'rb'))
        train_uncertainty_attr_labels = []
        for d in train_data:
            attr_certainty = np.array(d['attribute_certainty'])
            if len(mask) > len(attr_certainty):
                train_uncertainty_attr_labels.extend(list(attr_certainty[mask]))
            else:
                train_uncertainty_attr_labels.extend(attr_certainty)

        # evaluate the bottleneck model
        args.eval_data = f'train'
        train_class_labels, train_attr_labels, train_attr_outputs, train_attr_outputs_sigmoid, \
            train_encodings = eval_inference(args, use_encoding=True)

        train_attr_binary_outputs = np.rint(train_attr_outputs_sigmoid).astype(int)

        # compute model2 outputs
        train_attr_outputs = np.array(train_attr_outputs)
        stage2_input = torch.from_numpy(train_attr_outputs)
        stage2_input = stage2_input.reshape(-1, n_attributes).to(device)
        stage2_output = model2(stage2_input)
        _, preds = stage2_output.topk(1, 1)
        class_outputs = preds.data.cpu().numpy().squeeze()
        class_logits = stage2_output.cpu().detach().numpy()

        train_info = {
            'class_labels': np.array(train_class_labels),
            'attr_labels': np.array(train_attr_labels),
            'attr_outputs': train_attr_outputs,
            'attr_outputs_sig': np.array(train_attr_outputs_sigmoid),
            'attr_binary_outputs': train_attr_binary_outputs,
            'encodings': train_encodings,
            'uncertainty_attr_labels': np.array(train_uncertainty_attr_labels),
            'class_outputs': class_outputs,
            'class_logits': class_logits,
        }
        results.append(train_info)
    else:
        results.append(dict())

    if 'val' in data_splits:
        val_data = pickle.load(open(os.path.join(args.data_dir, f'val{fold}.pkl'), 'rb'))
        val_uncertainty_attr_labels = []
        for d in val_data:   # dropping the last elements might also be necessary here
            attr_certainty = np.array(d['attribute_certainty'])
            val_uncertainty_attr_labels.extend(list(attr_certainty[mask]))

        args.eval_data = f'val'
        val_class_labels, val_attr_labels, val_attr_outputs, val_attr_outputs_sigmoid, \
            val_encodings = eval_inference(args, use_encoding=True)

        val_attr_binary_outputs = np.rint(val_attr_outputs_sigmoid).astype(int)

        # compute model2 outputs
        val_attr_outputs = np.array(val_attr_outputs)
        stage2_input = torch.from_numpy(val_attr_outputs)
        stage2_input = stage2_input.reshape(-1, n_attributes).to(device)
        stage2_output = model2(stage2_input)
        _, preds = stage2_output.topk(1, 1)
        class_outputs = preds.data.cpu().numpy().squeeze()
        class_logits = stage2_output.cpu().detach().numpy()

        val_info = {
            'class_labels': np.array(val_class_labels),
            'attr_labels': np.array(val_attr_labels),
            'attr_outputs': val_attr_outputs,
            'attr_outputs_sig': np.array(val_attr_outputs_sigmoid),
            'attr_binary_outputs': val_attr_binary_outputs,
            'encodings': val_encodings,
            'uncertainty_attr_labels': np.array(val_uncertainty_attr_labels),
            'class_outputs': class_outputs,
            'class_logits': class_logits,
        }
        results.append(val_info)
    else:
        results.append(dict())

    if 'test' in data_splits:
        try:
            test_data = pickle.load(open(os.path.join(args.data_dir, f'test{fold}.pkl'), 'rb'))
        except FileNotFoundError:
            test_data = pickle.load(open(os.path.join(args.data_dir, 'test.pkl'), 'rb'))
        test_uncertainty_attr_labels = []
        for d in test_data:
            attr_certainty = np.array(d['attribute_certainty'])
            test_uncertainty_attr_labels.extend(list(attr_certainty[mask]))

        args.eval_data = 'test'
        test_class_labels, test_attr_labels, test_attr_outputs, test_attr_outputs_sigmoid, \
            test_encodings = eval_inference(args, use_encoding=True, model_training=False)       # model training false?

        test_attr_binary_outputs = np.rint(test_attr_outputs_sigmoid).astype(int)

        # compute model2 outputs
        test_attr_outputs = np.array(test_attr_outputs)
        stage2_input = torch.from_numpy(test_attr_outputs)
        stage2_input = stage2_input.reshape(-1, n_attributes).to(device)
        stage2_output = model2(stage2_input)
        _, preds = stage2_output.topk(1, 1)
        class_outputs = preds.data.cpu().numpy().squeeze()
        class_logits = stage2_output.cpu().detach().numpy()

        test_info = {
            'class_labels': np.array(test_class_labels),
            'attr_labels': np.array(test_attr_labels),
            'attr_outputs': test_attr_outputs,
            'attr_outputs_sig': np.array(test_attr_outputs_sigmoid),
            'attr_binary_outputs': test_attr_binary_outputs,
            'encodings': test_encodings,
            'uncertainty_attr_labels': np.array(test_uncertainty_attr_labels),
            'class_outputs': class_outputs,
            'class_logits': class_logits,
        }
        results.append(test_info)
    else:
        results.append(dict())

    return results


def prepare_test_shift(args, model2, n_attributes):
    """
    Evaluates the model on distribution shifted test data. Returns a dict test_info which is structured the same way as
    the results of prepare data.
    """
    # load the attribute mask
    mask = pickle.load(open(os.path.join(args.data_dir, 'mask.pkl'), 'rb'))

    test_data = pickle.load(open(os.path.join(args.data_dir, 'test.pkl'), 'rb'))
    test_uncertainty_attr_labels = []
    for d in test_data:
        attr_certainty = np.array(d['attribute_certainty'])
        test_uncertainty_attr_labels.extend(list(attr_certainty[mask]))

    tmp_data_dir = args.data_dir

    # for the evaluation, change the data_dir parameter of args
    args.data_dir = f'CUB_processed/adversarial_{args.test_shift}'
    args.eval_data = 'test'
    test_class_labels, test_attr_labels, test_attr_outputs, test_attr_outputs_sigmoid,\
        test_encodings = eval_inference(args, use_encoding=True)

    test_attr_binary_outputs = np.rint(test_attr_outputs_sigmoid).astype(int)

    # compute model2 outputs
    test_attr_outputs = np.array(test_attr_outputs)
    stage2_input = torch.from_numpy(test_attr_outputs)
    stage2_input = stage2_input.reshape(-1, n_attributes).to(device)
    stage2_output = model2(stage2_input)
    _, preds = stage2_output.topk(1, 1)
    class_outputs = preds.data.cpu().numpy().squeeze()
    class_logits = stage2_output.cpu().detach().numpy()

    test_info = {
        'class_labels': np.array(test_class_labels),
        'attr_labels': np.array(test_attr_labels),
        'attr_outputs': test_attr_outputs,
        'attr_outputs_sig': np.array(test_attr_outputs_sigmoid),
        'attr_binary_outputs': test_attr_binary_outputs,
        'encodings': test_encodings,
        'uncertainty_attr_labels': np.array(test_uncertainty_attr_labels),
        'class_outputs': class_outputs,
        'class_logits': class_logits,
    }

    args.data_dir = tmp_data_dir
    return test_info


def store_prepared_data(info, name, dataset):
    """
    Stores precomputed train, val and test info to pickle files to avoid recomputing them.
    Folder: precomputed
    filename: name_info.pkl
    """
    os.makedirs(f'precomputed/{dataset}/', exist_ok=True)
    with open(f'precomputed/{dataset}/{name}_info.pkl', 'wb') as f:
        pickle.dump(info, f)


def load_prepared_data(name, dataset):
    """
    Loads precomputed information about train, val, test or shifted splits from the corresponding pickle files.
    Folder: precomputed
    filename: name_info.pkl
    """
    with open(f'precomputed/{dataset}/{name}_info.pkl', 'rb') as f:
        data = pickle.load(f)
    return data


def setup(data_info, memory_module, attr_group_dict, threshold):
    """
    Fill the memory module with data from the given split.
    Parameters:
        data_info: information about the validation split, provided by prepare_data
        memory_module: the nearest neighbor module to fill.
        attr_group_dict: dictionary containing a mapping from concept groups to concepts
        threshold: maximum accuracy on the validation set for an example to be added to the retriever
    """
    n_attributes = int(len(data_info['attr_labels']) / len(data_info['class_labels']))

    img_ids = range(len(data_info['class_labels']))
    attr_check = data_info['attr_labels'] == data_info['attr_binary_outputs']
    cl_check = data_info['class_labels'] == data_info['class_outputs']

    for img_id in img_ids:
        attr_acc = np.mean(attr_check[img_id * n_attributes: (img_id + 1) * n_attributes])
        if attr_acc < threshold and not cl_check[img_id]:
            order = lcp(data_info['attr_outputs_sig'][img_id * n_attributes: (img_id + 1) * n_attributes],
                        attr_group_dict,
                        data_info['attr_labels'][img_id * n_attributes: (img_id + 1) * n_attributes])
            key = data_info['encodings'][img_id]

            memory_module.add_intervention(key, order)


def select_intervention_instances(data_info, memory_module, threshold):
    """
    Select the instances to perform interventions on.
    Parameters:
        data_info: information about the data where instances should be selected
        memory_module: nearest neighbor module
        threshold: Distance threshold to select whether an example should be included or not
    Returns:
        List of img indices (based on the order of data_info), for intervening
    """
    img_ids = list(range(len(data_info['class_labels'])))
    intervention_ids = []

    for img_id in img_ids:
        key = data_info['encodings'][img_id]

        if memory_module.check_intervention(key, threshold):
            # this image has been selected for intervention:
            intervention_ids.append(img_id)

    return intervention_ids


def select_intervention_instances_perfect(data_info):
    """
    Returns the indices of all test instances with wrong class prediction.
    Parameters:
        data_info: information about the data where instances should be selected
    Returns:
        List of img indices (based on the order of data_info), for intervening
    """
    compare = data_info['class_labels'] == data_info['class_outputs']
    intervention_ids = np.nonzero(~compare)[0]

    return intervention_ids.tolist()


def select_interventions(intervention_ids, data_info, attr_group_dict, method, n_concepts, ectp_info=None):
    """
    Select the instances (and concepts) to perform interventions on.
    Parameters:
        intervention_ids: Ids of images selected for intervention.
        data_info: information about the data
        attr_group_dict: mapping between attributes and groups
        method: which method to use to select the concepts for the examples. One of: 'rand', 'ucp', 'lcp', 'ectp'.
        n_concepts: Number of concepts to retrieve for each example
        ectp_info: Precomputed information for the method ectp. Tuple (mod_0_output, mod_1_output) where mod_0_output
            is model2 output where all attributes have been set to 0 (individually) and mod_1_output is the same if
            they have been set to 1.

    Returns:
        List of concept indices (based on data_info), indicating absolute position of concepts in the list for
            interventions.
    """
    all_global_concepts = []

    n_attributes = int(len(data_info['attr_labels']) / len(data_info['class_labels']))

    for img_id in intervention_ids:

        # get concept group indices (without reapplying attr_group mapping)
        predictions = data_info['attr_outputs'][img_id * n_attributes: (img_id + 1) * n_attributes]
        if method == 'rand':
            local_concepts = rand(predictions, attr_group_dict)
        elif method == 'ucp':
            local_concepts = ucp(predictions, attr_group_dict)
        elif method == 'ectp':
            assert ectp_info is not None
            # extract ectp info
            out_0 = []
            out_1 = []
            for i in range(n_attributes):
                out_0.append(ectp_info[0][i][img_id])
                out_1.append(ectp_info[1][i][img_id])

            local_concepts = ectp(predictions, data_info['class_logits'][img_id], attr_group_dict, out_0, out_1)
        elif method == 'lcp':
            ground_truth = data_info['attr_labels'][img_id * n_attributes: (img_id + 1) * n_attributes]
            local_concepts = lcp(predictions, attr_group_dict, ground_truth)
        else:
            raise ValueError(f"Specified method is not valid: {method}. Select one of rand, ucp, ectp, lcp.")

        # at most n_concepts for intervention
        local_concepts = local_concepts[:n_concepts]

        # change grouping back to individual concepts:
        local_concepts = [c for i in local_concepts for c in attr_group_dict[i]]

        # update local concept idx to global idx
        global_concepts = (np.array(local_concepts) + img_id * n_attributes).tolist()

        all_global_concepts.extend(global_concepts)

    return all_global_concepts


def fill_memory(intervention_ids, data_info, attr_group_dict, memory_module, ectp_info, use_invisible):
    """
    Fill the memory with interventional data (i.e. human "ground-truth" interventions).
    Parameters:
        intervention_ids: Ids of images selected for intervention.
        data_info: information about the data
        attr_group_dict: mapping between attributes and groups
        memory_module: memory module to fill (its internal data will be updated)
        ectp_info: Precomputed information for the method ectp. Tuple (mod_0_output, mod_1_output) where mod_0_output
            is model2 output where all attributes have been set to 0 (individually) and mod_1_output is the same if
            they have been set to 1.
        use_invisible: Whether to use invisible concept for interventions or not.
    """
    n_attributes = int(len(data_info['attr_labels']) / len(data_info['class_labels']))

    not_visible_idx = np.where(data_info['uncertainty_attr_labels'] == 1)[0]
    for img_id in intervention_ids:
        # order the concepts with ectp, record interventions for them
        predictions = data_info['attr_outputs'][img_id * n_attributes: (img_id + 1) * n_attributes]
        out_0 = []
        out_1 = []
        for i in range(n_attributes):
            out_0.append(ectp_info[0][i][img_id])
            out_1.append(ectp_info[1][i][img_id])

        # intervention index on groups:
        inter_group_idx = ectp(predictions, data_info['class_logits'][img_id], attr_group_dict, out_0, out_1).tolist()
        # change grouping back to individual concepts:
        inter_attr_idx = [c for i in inter_group_idx for c in attr_group_dict[i]]
        group_visible = []

        # handle not visible attributes
        if not use_invisible:
            # check all groups
            for idx in inter_group_idx:
                # count the number of visible attributes per group
                num_vis = 0
                for attr_idx in attr_group_dict[idx]:
                    if attr_idx + img_id * n_attributes in not_visible_idx:
                        # remove all not visible attributes from the attribute index
                        inter_attr_idx.remove(attr_idx)
                    else:
                        num_vis += 1

                # if at least some of the attributes of the group are visible, keep it
                if num_vis > 0:
                    group_visible.append(num_vis)

        # get ground truth values for the intervened concepts:
        ground_truth = data_info['attr_labels'][img_id * n_attributes: (img_id + 1) * n_attributes]
        inter_values = ground_truth[inter_attr_idx]

        intervention = (group_visible, inter_attr_idx, inter_values)

        # add intervention to the retriever
        key = data_info['encodings'][img_id]
        memory_module.add_intervention(key, intervention)


def evaluate_interventions(data_info, img_ids, global_interventions, model2, use_invisible):
    """
    Given parameters, method and selected interventions:
    Evaluate the instances selected for interventions by computing the following metrics:
        Class_accuracy      after interventions (on the instances with interventions as well as on all instances)
        Attribute_accuracy  before interventions (attribute acc after interventions is always 100%, except invisible)
    To compute this metric, evaluation by model2 are done.

    Parameters:
        data_info: information about the data
        img_ids: ids of all images which receive an intervention
        global_interventions: list of attribute indices for intervention (global indices, i.e. based on a flat view)
        model2: Predictor (c -> y) network to obtain the class output after interventions
        use_invisible: Whether to use invisible concept for interventions or not.
    Returns:
        Results as dict with keys: attr_acc_bef, cl_acc_aft, cl_acc_aft_tot
    """
    n_attributes = int(len(data_info['attr_labels']) / len(data_info['class_labels']))

    # skip interventions on invisible elements
    if not use_invisible:
        not_visible_idx = np.where(data_info['uncertainty_attr_labels'] == 1)[0]
        global_interventions = [idx for idx in global_interventions if idx not in not_visible_idx]

    # apply interventions on the selected attributes
    attr_with = np.copy(data_info['attr_outputs'])
    attr_with[global_interventions] = data_info['attr_labels'][global_interventions]

    # compute model2 output
    attr_with = attr_with.reshape(-1, n_attributes)
    stage2_input = torch.from_numpy(attr_with).to(device)
    stage2_output = model2(stage2_input)
    _, preds = stage2_output.topk(1, 1)
    class_outputs_int = preds.data.cpu().numpy().squeeze()

    # ======== Compute metrics ========
    compare_after = class_outputs_int == data_info['class_labels']

    cl_acc_after_total = np.mean(compare_after)
    cl_acc_after = np.mean(compare_after[img_ids])

    compare_attr = data_info['attr_labels'] == data_info['attr_binary_outputs']
    attr_acc_before = np.mean(compare_attr[global_interventions])

    return {'attr_acc_bef': attr_acc_before, 'cl_acc_aft': cl_acc_after, 'cl_acc_aft_tot': cl_acc_after_total}


def evaluate_generalization(memory_module, data_info, model2, threshold, n_concepts):
    """
    Given a filled CIR: Perform interventions on the given test data.
    Iterate over the test data and check for each example, if it receives an intervention.

    Evaluate the performance of the test set with retrieved interventions:
        Number of performed interventions
        Class accuracy: before and after interventions, as well as after interventions on the full dataset
        Concept accuracy: before and after interventions, as well as after interventions on the full dataset

    Parameters:
        memory_module: memory_module filled with human interventions
        data_info: information about the data
        model2: Predictor (c -> y) network to obtain the class output after interventions
        threshold: threshold for distance to apply interventions
        n_concepts: number of concepts to apply interventions for
    Returns:
        Results as dict with keys: num_int, cl_acc_bef, cl_acc_aft, cl_acc_aft_tot, attr_acc_bef, attr_acc_aft
        attr_acc_aft_tot
    """
    n_attributes = int(len(data_info['attr_labels']) / len(data_info['class_labels']))
    img_ids = range(len(data_info['class_labels']))
    # with interventions:
    attr_with = np.copy(data_info['attr_outputs'])
    attr_with_binary = np.copy(data_info['attr_binary_outputs'])
    intervention_img_ids = []
    global_interventions = []

    for img_id in img_ids:
        # check for each image if there is an intervention, if so: apply it
        img_key = data_info['encodings'][img_id]
        intervention = memory_module.get_intervention(img_key, threshold)
        if intervention is not None:
            # number of visible attributes in n_concept groups:
            group_visible = sum(intervention[0][:n_concepts])
            # update attribute idx and values to the number of concept groups selected
            inter_attr_idx = intervention[1][:group_visible]
            inter_val = intervention[2][:group_visible]

            # apply the intervention
            attr_with[img_id * n_attributes: (img_id + 1) * n_attributes][inter_attr_idx] = inter_val
            attr_with_binary[img_id * n_attributes: (img_id + 1) * n_attributes][inter_attr_idx] = inter_val

            # keep track of all intervened concept ids and img ids
            global_interventions.extend(list(np.array(inter_attr_idx) + img_id * n_attributes))
            intervention_img_ids.append(img_id)

    # compute model2 output (with interventions)
    attr_with = attr_with.reshape(-1, n_attributes)
    stage2_input = torch.from_numpy(attr_with).to(device)
    stage2_output = model2(stage2_input)
    _, preds = stage2_output.topk(1, 1)
    class_outputs_int = preds.data.cpu().numpy().squeeze()

    # ======== Compute metrics ========
    compare_cl_before = data_info['class_outputs'] == data_info['class_labels']
    compare_cl_after = class_outputs_int == data_info['class_labels']

    cl_acc_before = np.mean(compare_cl_before[intervention_img_ids])
    cl_acc_after = np.mean(compare_cl_after[intervention_img_ids])
    cl_acc_after_total = np.mean(compare_cl_after)

    compare_attr_before = data_info['attr_binary_outputs'] == data_info['attr_labels']
    compare_attr_after = attr_with_binary.flatten() == data_info['attr_labels']

    attr_acc_before = np.mean(compare_attr_before[global_interventions])
    attr_acc_after = np.mean(compare_attr_after[global_interventions])
    attr_acc_after_total = np.mean(compare_attr_after)
    return {'num_int': len(intervention_img_ids),
            'cl_acc_bef': cl_acc_before, 'cl_acc_aft': cl_acc_after, 'cl_acc_aft_tot': cl_acc_after_total,
            'attr_acc_bef': attr_acc_before, 'attr_acc_aft': attr_acc_after, 'attr_acc_aft_tot': attr_acc_after_total}
