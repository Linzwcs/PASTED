from typing import Any
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    DataCollatorForTokenClassification,
)
from nltk.tokenize import sent_tokenize
import torch
import pandas as pd
from tqdm import tqdm
from utils import jload
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import argparse
import numpy as np


class Detector:
    def __init__(self, model_name, device="cuda") -> None:
        # device = "cpu"  # use 'cuda:0' if GPU is available
        # self.model_dir = "nealcly/detection-longformer"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device

        self.model.to(device)
        self.model.eval()

    def detect(self, input_text, th=-3.08583984375):

        tokenizer, model, device = self.tokenizer, self.model, self.device
        label2decisions = {
            0: "machine-generated",
            1: "human-written",
        }
        tokenize_input = tokenizer(input_text)
        tensor_input = torch.tensor([tokenize_input["input_ids"]]).to(device)
        outputs = model(tensor_input)
        is_machine = -outputs.logits[0][0].item()

        if is_machine < th:
            decision = 0
        else:
            decision = 1
        return label2decisions[decision]


class Defender:
    def __init__(self, model_name, device):
        if "classification" in model_name:
            num_labels = 2
        elif "multi-dimension" in model_name:
            num_labels = 3
        else:
            num_labels = 1
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device

        self.model.to(device)
        self.model.eval()

    def __call__(self, text, threshold=None):

        sents = sent_tokenize(text)
        text = " </s> ".join(sents)

        input_ids = self.tokenizer(
            text, max_length=2048, truncation=True, padding="max_length"
        )["input_ids"]
        sent_label_idx = [i for i, ids in enumerate(input_ids) if ids == 2]

        tensor_input = torch.tensor([input_ids]).to(self.device)
        outputs = self.model(tensor_input).logits.detach().cpu().numpy()
        outputs_logits = outputs[0][sent_label_idx]
        outputs_logits: np.ndarray
        if outputs_logits.shape[1] == 2:
            outputs_logits = outputs_logits[:, 1]
        outputs_logits = outputs_logits.flatten()
        if threshold is None:
            return outputs_logits.mean(axis=-1)
        else:
            return outputs_logits.mean(axis=-1) > threshold


def pipe(
    text,
    defender,
    detector,
    defender_threshold=0,
    detector_threshold=-3.08583984375,
):
    mean_label = defender(text, defender_threshold)
    if mean_label == 1:
        return "machine-generated"
    label = detector(text, detector_threshold)
    return label
