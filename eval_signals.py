from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize, sent_tokenize
from utils import jload, jdump
from tqdm import tqdm
import stanza
import numpy as np
from scipy.stats import pearsonr
import pandas as pd

from datasets import load_metric, Metric
from alignment.score_sent_pairwises import SimJudger
from sentence_transformers import SentenceTransformer
import torch
from nltk.tree import Tree
from zss import simple_distance, Node
import random

"""
static vars
"""
_stanza = stanza.Pipeline(
    "en", processors="tokenize,constituency,pos", download_method=None
)
_syntax_metric = load_metric("gold_metrics.py")
# _syntax_metric: Metric
_judger = SimJudger(
    SentenceTransformer("sentence-transformers/all-mpnet-base-v2"),
    device="cuda" if torch.cuda.is_available() else "cpu",
)

"""
functions
"""


def Classification_diversity(candidate: str, reference: str):
    return 1


def GLOD_diversity(candidate: str, reference: str):
    try:
        ans = _syntax_metric.compute(predictions=[candidate], references=[reference])
    except:
        return {"lexical_diversity": 1, "syntax_diversity": 1}
    lexical_diversity = ans["set_diversity"][0]
    syntax_diversity = ans["syn_diversity"][0]
    return {
        "lexical_diversity": lexical_diversity,
        "syntax_diversity": syntax_diversity,
    }


def SEMANTIC_diversity(candidate: str, reference: str):
    score = _judger.judge([candidate], [reference])
    return 1 / 2 - score[0] / 2


def SYNTAX_diversity(candidate: str, reference: str):
    candidate_tree = __syntax_parse(candidate)
    reference_tree = __syntax_parse(reference)
    size = max(
        __get_tree_size(candidate_tree),
        __get_tree_size(reference_tree),
    )
    return simple_distance(candidate_tree, reference_tree) / size


def POS_BLEU_1_diversity(candidate: str, reference: str):
    candidate_doc = _stanza(candidate)
    reference_doc = _stanza(reference)
    candidate, reference = [], []

    for _, sent in enumerate(candidate_doc.sentences):
        candidate += [w.pos for w in sent.words]
    for _, sent in enumerate(reference_doc.sentences):
        reference += [w.pos for w in sent.words]

    references = [reference]
    return 1 - sentence_bleu(references, candidate, weights=(1, 0, 0, 0))


def POS_BLEU_4_diversity(candidate: str, reference: str):
    candidate_doc = _stanza(candidate)
    reference_doc = _stanza(reference)
    candidate, reference = [], []

    for _, sent in enumerate(candidate_doc.sentences):
        candidate += [w.pos for w in sent.words]
    for _, sent in enumerate(reference_doc.sentences):
        reference += [w.pos for w in sent.words]

    references = [reference]
    return 1 - sentence_bleu(references, candidate)


def BLEU_1_diversity(candidate: str, reference: str):
    candidate = word_tokenize(candidate)
    references = [word_tokenize(reference)]
    return 1 - sentence_bleu(references, candidate, weights=(1, 0, 0, 0))


def BLEU_4_diversity(candidate: str, reference: str):
    candidate = word_tokenize(candidate)
    references = [word_tokenize(reference)]
    return 1 - sentence_bleu(references, candidate)


def __get_tree_size(tree):
    sub_tree_size = 0
    for child in tree.children:
        sub_tree_size += __get_tree_size(child)
    return 1 + sub_tree_size


def __syntax_parse(text: str, max_dep=3):
    root = Node("text-root")
    doc = _stanza(text)
    for sentence in doc.sentences:
        tree_str = str(sentence.constituency)
        tree = Tree.fromstring(tree_str)
        node = __convert_nltk2zss_tree(tree, max_dep=max_dep)
        if node is not None:
            node.label = "sent-root"
            root.addkid(node)
    return root


def __convert_nltk2zss_tree(tree, dep=0, max_dep=3):
    if dep > max_dep:
        return None
    elif type(tree) is str:
        return Node(tree)
    else:
        root = Node(tree.label())
        for tree_child in list(tree):
            child = __convert_nltk2zss_tree(tree_child, dep + 1, max_dep)
            if child is None:
                continue
            else:
                root.addkid(child)
        return root
