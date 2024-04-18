import argparse
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


class SimJudger:
    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def judge(self, para_sents, raw_sents):
        para_embs = self.model.encode(para_sents)
        raw_embs = self.model.encode(raw_sents)
        sim_matrix = util.pytorch_cos_sim(para_embs, raw_embs)
        return sim_matrix.detach().cpu().numpy().tolist()
