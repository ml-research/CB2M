import torch
import numpy as np
from scipy.special import softmax
from scipy.stats import entropy

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def grouping_and_order(scores, groupings):
    """
    Takes an array of scores and a grouping dictionary (or None).
    Groups the scores according to the dict and the computes an ordering based on the (group) scores.
    Concept groups are handled by average score, according to the closer paper.
    """
    # average the scores of groups if a grouping is given
    if groupings is not None:
        tmp_scores = np.zeros(len(groupings.keys()))
        for i in groupings.keys():
            tmp_sum = 0
            for j in groupings[i]:
                tmp_sum += scores[j]
            tmp_scores[i] = tmp_sum / len(groupings[i])
        scores = tmp_scores
    # order indices based on array values
    order = np.argsort(-scores)
    return order


def rand(predictions, groupings):
    """
    Random selection of concept order
    Every concept gets a random score.
    """
    score = np.random.rand(predictions.shape[0])
    return grouping_and_order(score, groupings)


def ucp(predictions, groupings):
    """
    (local) uncertainty of concept prediction.
    Entropy of the (binary) concept prediction.
    """
    score = 1 / np.abs(predictions - 0.5) ** 2
    return grouping_and_order(score, groupings)


def ectp(predictions, class_logits, groupings, out_0, out_1):
    """
    Expected change in target prediction.
    KL divergence between model output and model_output for concept set to 1;0 respectively, weighted by the
    model prediction of the concept being 1;0.
    :param class_logits: class logit output of the model with unmodified concepts
    :param out_0: model outputs if concepts are set to 0
    :param out_1: model outputs if concepts are set to 1
    """

    dkl_0 = np.zeros(predictions.shape)
    dkl_1 = np.zeros(predictions.shape)

    original_softmax = softmax(np.array(class_logits))

    for i in range(predictions.shape[0]):
        # prepare the modified model inputs
        class_outputs_0 = out_0[i]
        class_outputs_1 = out_1[i]

        softmax_outputs_0 = softmax(class_outputs_0)
        softmax_outputs_1 = softmax(class_outputs_1)

        dkl_0[i] = entropy(softmax_outputs_0, original_softmax)
        dkl_1[i] = entropy(softmax_outputs_1, original_softmax)

    score = (np.ones(predictions.shape) - predictions) * dkl_0 + predictions * dkl_1

    return grouping_and_order(score, groupings)


def ectp_precompute(test_info, n_attributes, model2):
    """
    Precompute the data with interventions for the ectp method.
    :param test_info: precompute information about the test set
    :param n_attributes: number of concepts
    :param model2: c -> y model
    :return: (cl_out_0, cl_out_1) - combined outputs if concepts were set to 0 or 1 respectively
    """
    class_out_0_combined = []
    class_out_1_combined = []
    for i in range(n_attributes):
        # generate modified prediction (set the attribute to 0 and 1 respectively)
        modified_pred_0 = torch.tensor(test_info['attr_outputs'])
        modified_pred_1 = torch.tensor(test_info['attr_outputs'])
        modified_pred_0 = modified_pred_0.reshape(-1, n_attributes)
        modified_pred_1 = modified_pred_1.reshape(-1, n_attributes)
        modified_pred_0[:, i] = 0
        modified_pred_1[:, i] = 1
        modified_pred_0 = modified_pred_0.to(device)
        modified_pred_1 = modified_pred_1.to(device)

        # evaluate model2 with different inputs
        class_outputs_0 = model2(modified_pred_0).cpu().detach().numpy()
        class_outputs_1 = model2(modified_pred_1).cpu().detach().numpy()
        class_out_0_combined.append(class_outputs_0)
        class_out_1_combined.append(class_outputs_1)

    return class_out_0_combined, class_out_1_combined


def lcp(predictions, groupings, ground_truth):
    """
    Loss on concept prediction (oracle)
    """
    score = np.abs(predictions - ground_truth) ** 2
    return grouping_and_order(score, groupings)
