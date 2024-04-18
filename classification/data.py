# from data import ParaDataGenerator
import os
from datasets import load_dataset
from torch.utils.data import DataLoader
import random
import numpy as np
import json
import evaluate
from sklearn.metrics import classification_report
import logging
import pandas as pd
from nltk import sent_tokenize
from sklearn.metrics import roc_auc_score
from sklearn import metrics

logger = logging.getLogger("__main__")
# seqeval = evaluate.combine(["accuracy", "f1", "precision", "recall"])
import pickle as plk


def compute_metrics_for_seq(p):
    predictions, labels = p
    predictions, labels = (
        predictions.reshape((-1, 2)).tolist(),
        labels.flatten().tolist(),
    )

    true_predictions_logits = np.array(
        [prediction for prediction, label in zip(predictions, labels) if label != -100],
        dtype=np.float32,
    )
    true_labels = np.array(
        [label for prediction, label in zip(predictions, labels) if label != -100],
        dtype=np.int32,
    )
    fpr, tpr, thresholds = metrics.roc_curve(
        true_labels, true_predictions_logits[:, 1], pos_label=1
    )
    # print(true_predictions_logits)
    # optimal_threshold_index = (tpr - fpr).argmax()
    # optimal_threshold = thresholds[optimal_threshold_index]
    ## print(fpr, tpr, thresholds)
    # true_predictions = true_predictions_logits[:, 1] > optimal_threshold

    with open("prediction.plk", "wb") as f:
        plk.dump(true_predictions_logits[:, 1], f)
    return {
        "metric": "classification",
        "Detection Accuracy": tpr[fpr < 0.01][-1],
        "auc": roc_auc_score(true_labels, true_predictions_logits[:, 1]),
        # "correlation_coefficient": correlation_coefficient,
        # "p_value":p_value
        # "auc": roc_auc_score(1-true_labels, predictions_logits),
    }
