# from data import ParaDataGenerator
import os

# from datasets import load_dataset
from torch.utils.data import DataLoader
import random
import numpy as np
import json
from scipy.stats import pearsonr
import evaluate
from sklearn.metrics import classification_report
import logging
import pandas as pd
from nltk import sent_tokenize
from sklearn.metrics import roc_auc_score
from sklearn import metrics


# import evaluate
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

logger = logging.getLogger("__main__")
# seqeval = evaluate.combine(["accuracy", "f1", "precision", "recall"])


def compute_metrics_for_diversity(p):
    # threshold = 0.03

    predictions, references = p
    # metrics={}
    predictions = predictions.squeeze(-1)
    masks = references >= 0

    references *= masks
    predictions *= masks

    mse = mean_squared_error(references, predictions)

    masks = (1 - masks).astype(np.int32).flatten()
    masks *= -100

    # prediction_logits = predictions.flatten().tolist()
    # labels = np.abs(references - 1) < 1e-4
    # predictions = (predictions < threshold).astype(np.int32)
    labels = np.abs(references) < 1e-6
    labels = 1 - labels
    # predictions= predictions
    labels = labels.flatten().tolist()
    predictions = predictions.flatten().tolist()
    # print(len(predictions),len(references))

    true_labels = np.array(
        [
            label
            for prediction, label, mask in zip(predictions, labels, masks)
            if mask != -100
        ],
        dtype=np.int32,
    )
    predictions_logits = np.array(
        [
            prediction_logit
            for prediction_logit, label, mask in zip(predictions, labels, masks)
            if mask != -100
        ],
    )

    # print(references.shape,masks.shape)
    references = np.array(
        [
            r
            for r, label, mask in zip(references.flatten(), labels, masks)
            if mask != -100
        ],
    )
    # print(references)
    # print("----------\n",len(references),len(predictions_logits))
    fpr, tpr, thresholds = metrics.roc_curve(
        true_labels, predictions_logits, pos_label=1
    )
    #
    # optimal_threshold_index = (tpr - fpr).argmax()
    # optimal_threshold = thresholds[optimal_threshold_index]
    # fpr[fpr < 0.01]
    # print(optimal_threshold)
    # print(true_labels)
    # print(predictions)
    # true_predictions = true_predictions_logits[:, 1] > optimal_threshold
    # true_predictions = (predictions_logits > optimal_threshold).astype(np.int32)
    np.save("prediction.npy", predictions_logits)
    # true_predictions = np.argmax(true_predictions_logits, axis=1)
    # print(true_labels.shape,true_predictions.shape)
    # return mse_metric.compute(
    #    predictions=predictions.view(-1), references=references.view(-1)
    # )
    # print(predictions.shape,references.shape)
    correlation_coefficient, p_value = pearsonr(predictions_logits, references)
    return {
        "metric": "diversity",
        "mse": mse,
        "Detection Accuracy": tpr[fpr < 0.01][-1],
        "auc": roc_auc_score(true_labels, predictions_logits),
        "correlation_coefficient": correlation_coefficient,
    }


#
# def save_json_by_lines(out_path, results, mode="a+"):
# with open(out_path, mode) as f:
#    for res in results:
#        res_str = json.dumps(res)
#        f.write(res_str + "\n")


#
#
# def load_json_by_lines(out_path):
#    data = []
#    with open(out_path, "r") as f:
#        for res in f.readlines():
#            res_json = json.loads(res)
#            data.append(res_json)
#    return data
#
#
# def create_sent_level_test_data(data, out_path):
#    with open(out_path, "w") as f:
#        for i, row in enumerate(data):
#            sents = sent_tokenize(row["text"])
#            f.write(
#                json.dumps(
#                    {
#                        "index": i,
#                        "text": " </s> ".join(sents),
#                        "src": row["src"],
#                        "label": [row["label"]] * len(sents),
#                        "is_para": None,
#                        # "sent_level": True
#                    }
#                )
#                + "\n"
#            )
#
#
# def create_input_data_files(data_files):
#    request_cols = ["index", "text", "src", "label", "is_para"]
#    for key in data_files:
#        file_path = data_files[key]
#        file_dir, file_name = os.path.split(file_path)
#        file_name, ext = "".join(file_name.split(".")[:-1]), file_name.split(".")[-1]
#        cache_dir = os.path.join(file_dir, "cache")
#        cache_path = os.path.join(cache_dir, file_name + ".json")
#        if not os.path.exists(cache_dir):
#            os.makedirs(cache_dir)
#        if ext == "csv":
#            df = pd.read_csv(file_path)
#        else:
#            df = pd.read_json(file_path, lines=True)
#        # print(df.columns.to_list(),request_cols,df.columns.to_list() == request_cols)
#        if set(df.columns.to_list()) == set(request_cols):
#            pass
#        else:
#            create_sent_level_test_data(df.to_dict(orient="records"), cache_path)
#            data_files[key] = cache_path
#    print(data_files)
#    return data_files
