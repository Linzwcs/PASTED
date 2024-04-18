from alignment.score_sent_pairwises import SimJudger, evaluate
from alignment.paraphrase_alignment import generate_pairwise
import argparse
import numpy as np
from utils import jload, jdump
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


def evaluate(args, judger: SimJudger):
    data = jload(args.raw_file)
    for item in tqdm(data):
        pairwise_dict = generate_pairwise(item)

        scores = judger.judge(
            pairwise_dict["para_sents"],
            pairwise_dict["raw_sents"],
        )
        shape = pairwise_dict["shape"]
        assert len(scores) == shape[0] * shape[1]
        jdump(
            {
                "index": item["index"],
                "pre_text": item["precede_text"],
                "raw_sents": pairwise_dict["raw_sents"],
                "fol_text": item["following_text"],
                "para_sents": pairwise_dict["para_sents"],
                "scores": scores,
                "shape": pairwise_dict["shape"],
                "src": item["src"],
            },
            args.out_file,
            mode="a",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", help="path to saved model")
    parser.add_argument(
        "--device", default="cuda", type=str, help="whether to train on gpu"
    )
    parser.add_argument("--raw-file", help="raw file")
    parser.add_argument("--out-file", help="out file")
    args = parser.parse_args()
    model = SentenceTransformer(args.model_path)
    judger = SimJudger(model, device=args.device)

    evaluate(args, model)
