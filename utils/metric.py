from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import sys
import logging
import numpy as np
from collections import Counter
from utils.const import misinfo_tasks

logger = logging.getLogger(__name__)

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, classification_report, accuracy_score, roc_auc_score

def get_statistics_about_correct_and_wrong(preds, labels):
    diff = [int(abs(pred[0] - pred[1])) for pred in preds]
    print("ALL")
    print("mean", np.mean(diff))
    print("median", np.median(diff))
    print(Counter(diff))

    preds_argmax = np.argmax(preds, axis=1)

    # if pred != label, save its pred and true label.
    false_bool_mask = (preds_argmax != labels)
    false_preds = preds[false_bool_mask]
    false_labels = labels[false_bool_mask]

    false_diff = [int(abs(pred[0] - pred[1])) for pred in false_preds]
    print("False")
    print("mean", np.mean(false_diff))
    print("median", np.median(false_diff))
    print(Counter(false_diff))

    true_bool = (preds_argmax == labels)
    true_preds = preds[true_bool]
    true_labels = labels[true_bool]

    true_diff = [int(abs(pred[0] - pred[1])) for pred in true_preds]
    print("True")
    print("mean", np.mean(true_diff))
    print("median", np.median(true_diff))
    print(Counter(true_diff))

def evaluate_true_results(preds, labels, batches, guids):
    filtered_preds, filtered_labels, filtered_batches = preds, labels, batches

    confidence = np.array(filtered_preds)
    diff = np.array([abs(pred[0] - pred[1]) for pred in filtered_preds])

    filtered_preds = np.argmax(filtered_preds, axis=1)
    true_bool_mask = (filtered_preds == filtered_labels)

    filtered_preds = np.array(filtered_preds)[true_bool_mask]
    filtered_labels = np.array(filtered_labels)[true_bool_mask]
    filtered_batches = np.array(filtered_batches)[true_bool_mask]
    filtered_confidence = confidence[true_bool_mask]
    filtered_diff = diff[true_bool_mask]
    filtered_guids = guids[true_bool_mask]

    positive_mask = (filtered_labels == 1)
    filtered_positive_preds = np.array(filtered_preds)[positive_mask]
    filtered_positive_labels = np.array(filtered_labels)[positive_mask]
    filtered_positive_batches = np.array(filtered_batches)[positive_mask]
    filtered_positive_guids = np.array(filtered_guids)[positive_mask]
    filtered_positive_confidence = filtered_confidence[positive_mask][:, 1]
    filtered_pos_diff = filtered_diff[positive_mask]

    negative_mask = (filtered_labels == 0)
    filtered_negative_preds = np.array(filtered_preds)[negative_mask]
    filtered_negative_labels = np.array(filtered_labels)[negative_mask]
    filtered_negative_batches = np.array(filtered_batches)[negative_mask]
    filtered_negative_guids = np.array(filtered_guids)[negative_mask]
    filtered_negative_confidence = filtered_confidence[negative_mask][:, 0]
    filtered_neg_diff = filtered_diff[negative_mask]

    return ({"preds": filtered_positive_preds, "labels": filtered_positive_labels, "batches": filtered_positive_batches,
             "diff": filtered_pos_diff, "guids": filtered_positive_guids},
            {"preds": filtered_negative_preds, "labels": filtered_negative_labels, "batches": filtered_negative_batches,
             "diff": filtered_neg_diff, "guids": filtered_negative_guids})


def evaluate_false_results(preds, labels, batches, guids):
    filtered_preds, filtered_labels, filtered_batches = preds, labels, batches

    confidence = np.array(filtered_preds)
    diff = np.array([abs(pred[0] - pred[1]) for pred in filtered_preds])

    filtered_preds = np.argmax(filtered_preds, axis=1)
    false_bool_mask = (filtered_preds != filtered_labels)

    filtered_preds = np.array(filtered_preds)[false_bool_mask]
    filtered_labels = np.array(filtered_labels)[false_bool_mask]
    filtered_batches = np.array(filtered_batches)[false_bool_mask]
    filtered_confidence = confidence[false_bool_mask]
    filtered_diff = diff[false_bool_mask]
    filtered_guids = guids[false_bool_mask]

    positive_mask = (filtered_labels == 0)
    filtered_positive_preds = np.array(filtered_preds)[positive_mask]
    filtered_positive_labels = np.array(filtered_labels)[positive_mask]
    filtered_positive_batches = np.array(filtered_batches)[positive_mask]
    filtered_positive_guids = np.array(filtered_guids)[positive_mask]
    filtered_positive_confidence = filtered_confidence[positive_mask][:, 1]
    filtered_pos_diff = filtered_diff[positive_mask]
    # filtered_pos_label_texts = filtered_label_text[positive_mask]

    negative_mask = (filtered_labels == 1)
    filtered_negative_preds = np.array(filtered_preds)[negative_mask]
    filtered_negative_labels = np.array(filtered_labels)[negative_mask]
    filtered_negative_batches = np.array(filtered_batches)[negative_mask]
    filtered_negative_guids = np.array(filtered_guids)[negative_mask]
    filtered_negative_confidence = filtered_confidence[negative_mask][:, 0]
    filtered_neg_diff = filtered_diff[negative_mask]
    # filtered_neg_label_texts = filtered_label_text[negative_mask]


    return ({"preds": filtered_positive_preds, "labels": filtered_positive_labels, "batches": filtered_positive_batches,
            "diff": filtered_pos_diff, "guids": filtered_positive_guids},
            {"preds": filtered_negative_preds, "labels": filtered_negative_labels, "batches": filtered_negative_batches,
             "diff": filtered_neg_diff, "guids": filtered_negative_guids})



def get_wrong_results(preds, labels, batches):
    preds = np.argmax(preds, axis=1)

    # if pred != label, save its pred and true label.
    bool_mask = (preds != labels)
    preds = preds[bool_mask]
    labels = labels[bool_mask]
    batches = batches[bool_mask]

    return {"preds": preds, "labels": labels, "batches": batches}

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    wegithed_f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
    macro_f1 = f1_score(y_true=labels, y_pred=preds, average='macro')

    return {
        "acc": acc,
        "weighted_f1": wegithed_f1,
        "macro_f1": macro_f1,
        "acc_and_f1": (acc + wegithed_f1) / 2,
    }

def all_standard_metric(preds, labels):
    # auc = roc_auc_score(y_true=labels, y_score=[p[1] for p in preds])
    acc = accuracy_score(y_true=labels, y_pred=preds)
    wegithed_f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
    macro_f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    report_dict = classification_report(y_true=labels, y_pred=preds, output_dict=True)

    # "auc": auc,
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "report": report_dict
    }

def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name in misinfo_tasks:
        return all_standard_metric(preds, labels)
    elif task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)
