# from data import ParaDataGenerator
import os

# from datasets import load_dataset
from torch.utils.data import DataLoader
import random
import numpy as np
import json
import torch
import evaluate
from sklearn.metrics import classification_report
import logging
import pandas as pd
from nltk import sent_tokenize
from sklearn.metrics import roc_auc_score

# import evaluate
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

logger = logging.getLogger("__main__")
# seqeval = evaluate.combine(["accuracy", "f1", "precision", "recall"])


def compute_metrics_for_moe(p):
    # threshold = 0.03

    predictions, references = p
    # metrics={}
    # predictions = predictions.squeeze(-1)
    masks = references >= 0

    references *= masks
    predictions *= masks

    lexical_mse = mean_squared_error(references[:, :, 0], predictions[:, :, 0])
    syntax_mse = mean_squared_error(references[:, :, 1], predictions[:, :, 1])
    semantic_mse = mean_squared_error(references[:, :, 2], predictions[:, :, 2])

    masks = (1 - masks).astype(np.int32).reshape((-1, 3))
    masks *= -100
    masks = masks[:, 0]

    prediction_logits = predictions.reshape((-1, 3))
    references = references.reshape((-1, 3))
    # predictions = (predictions > threshold).astype(np.int32)
    # labels = np.abs(references - 1) < 1e-4
    # predictions = (predictions < threshold).astype(np.int32)
    references: torch.Tensor
    labels = np.abs(references.sum(axis=-1)) < 1e-10
    labels = 1 - labels
    # predictions= predictions
    labels = labels.flatten().tolist()
    # predictions = predictions.flatten().tolist()
    # print(len(predictions),len(references))
    true_labels = np.array(
        [label for label, mask in zip(labels, masks) if mask != -100],
        dtype=np.int32,
    )
    predictions_logits = np.array(
        [
            prediction_logit.tolist()
            for prediction_logit, mask in zip(prediction_logits, masks)
            if mask != -100
        ],
    )

    assert prediction_logits.shape[1] == 3

    # true_predictions = np.argmax(true_predictions_logits, axis=1)
    # print(true_labels.shape,true_predictions.shape)
    # acc = (true_predictions == true_labels).sum() / len(true_labels)
    # machine_num = (true_labels == 0).sum()
    # human_num = (true_labels == 1).sum()
    # machine_recall = ((true_predictions == 0) * (true_labels == 0)).sum() / machine_num
    # human_recall = ((true_predictions == 1) * (true_labels == 1)).sum() / human_num

    # return mse_metric.compute(
    #    predictions=predictions.view(-1), references=references.view(-1)
    # )
    # print(predictions.shape,references.shape)
    return {
        "metric": "diversity",
        "lexical_mse": lexical_mse,
        "syntax_mse": syntax_mse,
        "semantic_mse": semantic_mse,
        "auc_dim_0": roc_auc_score(true_labels, predictions_logits[:, 0]),
        "auc_dim_1": roc_auc_score(true_labels, predictions_logits[:, 1]),
        "auc_dim_2": roc_auc_score(true_labels, predictions_logits[:, 2]),
        # "auc": roc_auc_score(1-true_labels, predictions_logits),
    }
