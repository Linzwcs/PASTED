#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
import pickle
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
from data import (
    compute_metrics_for_diversity,
)
import datasets
import numpy as np
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
import torch.nn.functional as F

import transformers
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
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from transformers import Trainer
import torch

os.environ["CURL_CA_BUNDLE"] = ""
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)

tasks = [
    "regression-bleu4",
    "regression-pos-bleu4",
    "regression-syntax",
]

logger = logging.getLogger(__name__)


class RegressionTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        """
        mask the token that is not "</s>" with 0, so that we can only calculate
        the loss on "</s>" token.
        """
        masks = (labels >= 0).float()
        labels = torch.mul(labels, masks)
        logits = torch.mul(logits, masks)

        return (
            (F.mse_loss(logits, labels), outputs)
            if return_outputs
            else F.mse_loss(logits, labels)
        )


class DataCollatorForTokenRegression(DataCollatorForTokenClassification):
    """
    Because "DataCollatorForTokenClassification" will transform the label
    type to int, so we slightly revise this class to fit our need.
    """

    def torch_call(self, features):
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )

        no_labels_features = [
            {k: v for k, v in feature.items() if k != label_name}
            for feature in features
        ]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                to_list(label)
                + [self.label_pad_token_id] * (sequence_length - len(label))
                for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label))
                + to_list(label)
                for label in labels
            ]

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.float)
        return batch


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(tasks)},
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )
    from_scratch: bool = field(
        default=False,
        metadata={"help": "set true to not load weights from pretrained models."},
    )
    # do_eval: Optional[bool] = field(
    #     default=False, metadata={"help": "do evaluation."}
    # )

    def __post_init__(self):
        assert self.task_name is not None
        self.task_name = self.task_name.lower()
        if self.task_name not in tasks:
            raise ValueError("Unknown task, you should pick one in " + ",".join(tasks))


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.validation_file == data_args.test_file:
        # training_args.do_eval = False
        pass
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    # training_args["report_to"] = None # disable integrations
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    logger.info(f"load task name: {data_args.task_name}")
    raw_datasets = load_dataset(data_args.dataset_name, data_args.task_name)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels

    num_labels = 1  # for regresssion
    # num_labels = 2 #for classification
    config = AutoConfig.from_pretrained(
        (
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path
        ),
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if not data_args.from_scratch:
        model = AutoModelForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=True,
        )
    else:
        model = AutoModelForTokenClassification.from_config(
            config=config,
            # ignore_mismatched_sizes=True,
        )
    # Preprocessing the raw_datasets
    # sentence1_key, sentence2_key = "text", None
    # if data_args.task_name is not None:
    #     sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    # else:
    #     # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
    #     non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
    #     if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
    #         sentence1_key, sentence2_key = "sentence1", "sentence2"
    #     else:
    #         if len(non_label_column_names) >= 2:
    #             sentence1_key, sentence2_key = non_label_column_names[:2]
    #         else:
    #             sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.\
    # label_to_id = None
    # if (
    #     model.config.label2id != PretrainedConfig(
    #         num_labels=num_labels).label2id
    #     and data_args.task_name is not None
    #     and not is_regression
    # ):
    #     # Some have all caps in their config, some don't.
    #     label_name_to_id = {
    #         k.lower(): v for k, v in model.config.label2id.items()}
    #     if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
    #         label_to_id = {
    #             i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
    #     else:
    #         logger.warning(
    #             "Your model seems to have been trained with labels, but they don't match the dataset: ",
    #             f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
    #             "\nIgnoring the model labels as a result.",
    #         )
    # elif data_args.task_name is None and not is_regression:
    #     label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            padding=padding,
            max_length=max_seq_length,
            truncation=True,
        )
        input_ids = tokenized_inputs["input_ids"]
        labels = []
        for i, label in enumerate(examples["label"]):
            word_ids = np.array(
                input_ids[i],
            )
            # token_id == 2 means this token is '</s>', which is our seperator between sentences
            label_idxs = np.where(word_ids == 2)[0]

            aligned_label = np.full(word_ids.shape, -100, dtype=np.float64)

            assert len(aligned_label.shape) == 1

            if len(label_idxs) != len(label):
                label = label[: len(label_idxs)]

            for idx, sub_label in zip(label_idxs, label):
                # we assign each "</s>" token with its corresponding label
                aligned_label[idx] = sub_label

            labels.append(aligned_label.tolist())

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            tokenize_and_align_labels,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        raw_datasets = raw_datasets.remove_columns(["label"])

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if (
            "validation" not in raw_datasets
            and "validation_matched" not in raw_datasets
        ):
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(
                range(data_args.max_predict_samples)
            )

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if data_args.pad_to_max_length:
        data_collator = DataCollatorForTokenRegression(tokenizer=tokenizer)
    elif training_args.fp16:
        data_collator = DataCollatorForTokenRegression(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    # print(type(data_collator(eval_dataset)))
    # Initialize our Trainer

    # print(train_dataset, eval_dataset)
    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics_for_diversity,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        # if data_args.task_name == "mnli":
        #     tasks.append("mnli-mm")
        #     eval_datasets.append(raw_datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [predict_dataset]

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=predict_dataset)

            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub:
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "tasks": "token-classification",
        }
        if data_args.task_name is not None:
            kwargs["language"] = "en"
            kwargs["dataset_tags"] = "glue"
            kwargs["dataset_args"] = data_args.task_name
            kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

        trainer.push_to_hub(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
