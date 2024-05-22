from detect import Detector
import argparse
from utils import jload
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import pickle as pkl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="path to saved model")
    parser.add_argument(
        "--device", default="cuda", type=str, help="whether to train on gpu"
    )
    parser.add_argument(
        "--golden_testfile",
        help="test set with golden syntactic and lexical signals",
    )
    args = parser.parse_args()
    detector = Detector(args.model_path, args.device)

    with open(args.golden_testfile, "rb") as f:
        data = pkl.load(f)
    syntactic_label, lexical_label, labels, logits = [], [], [], []

    for item in tqdm(data):
        predictions = detector(item["text"], False)
        out = [p[1] for p in predictions]
        logits += out
        syntactic_label += item["syntactic"][: len(out)]
        lexical_label += item["lexical"][: len(out)]
        labels += item["label"][: len(out)]
    syntactic_corr, _ = pearsonr(logits, syntactic_label)
    lexical_corr, _ = pearsonr(logits, lexical_label)
    fpr, tpr, _ = roc_curve(labels, logits, pos_label=1)

    metrics = {
        "syntactic_corr": syntactic_corr,
        "lexical_corr": lexical_corr,
        "auc": roc_auc_score(labels, logits),
        "Detection Accuracy": tpr[fpr < 0.01][-1],
    }

    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
