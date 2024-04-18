from nltk.tokenize import sent_tokenize
from itertools import product
import numpy as np
import numpy as np


def generate_pairwise(item):
    para_sents = sent_tokenize(item["para_text"])
    raw_sents = sent_tokenize(item["candidate_para_text"])
    sent_pairwise = product(para_sents, raw_sents)
    # sent_idx_pairwise = product(range(len(para_sents)), range(len(raw_sents)))
    return {
        "sent_pairwise": sent_pairwise,
        "para_sents": para_sents,
        "raw_sents": raw_sents,
        "shape": (len(para_sents), len(raw_sents)),
    }


def multi_flexible_alignment(similarity_matrix: np.ndarray, threshold):
    """Find alignments allowing multiple sentences to align to a single sentence."""
    n, _ = similarity_matrix.shape

    def calculate_score_spans(scores: np.ndarray):
        m = len(scores)
        score_spans = []
        for w_size in range(1, m + 1):
            for idx in range(m - w_size + 1):
                score_spans.append(
                    (
                        scores[idx : idx + w_size].mean(),
                        (idx, idx + w_size),
                        w_size,
                    )
                )
        return score_spans

    alignments = []
    for i in range(n):
        max_similarity = similarity_matrix[i].max()
        if max_similarity <= threshold:
            most_sim_idx = np.argmax(similarity_matrix[i])
            alignments.append((i, (most_sim_idx, most_sim_idx + 1)))
        else:
            score_spans = calculate_score_spans(similarity_matrix[i])
            avalibles = list(filter(lambda x: x[0] > threshold, score_spans))
            avalibles.sort(key=lambda x: x[2], reverse=True)
            alignments.append((i, avalibles[0][1]))

    return alignments


def alignment(item, threshold=0.7):
    # print(np.array(item["scores"]).reshape(item["shape"]))
    matrix = np.array(item["scores"]).reshape(item["shape"])  # >threshold
    para_sents, raw_sents = item["para_sents"], item["raw_sents"]

    align_idxs = multi_flexible_alignment(matrix, threshold=threshold)

    aligned_text = []
    for i, (start, end) in align_idxs:
        aligned_text.append(
            (para_sents[i].strip(), " ".join(raw_sents[start:end]).strip())
        )

    return aligned_text
