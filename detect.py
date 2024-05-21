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
        """
        return_type: sentence or text
        """
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
        elif outputs_logits.shape[1] == 3:
            outputs_logits = outputs_logits.mean(axis=-1)
        outputs_logits = outputs_logits.flatten()
        if threshold is None:
            return list(zip(sents, outputs_logits.tolist()))
        else:
            return list(zip(sents, (outputs_logits > threshold).tolist()))
