import torch
import numpy as np
from utils.reranking import re_ranking
from .rank import evaluate_rank
import torch.nn.functional as F
from config import cfg
#
@torch.no_grad()
def compute_euclidean_distance(features, others):
    m, n = features.size(0), others.size(0)
    dist_m = (
            torch.pow(features, 2).sum(dim=1, keepdim=True).expand(m, n)
            + torch.pow(others, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    dist_m.addmm_(1, -2, features, others.t())

    return dist_m.cpu().numpy()


@torch.no_grad()
def compute_cosine_distance(features, others):
    """Computes cosine distance.
    Args:
        features (torch.Tensor): 2-D feature matrix.
        others (torch.Tensor): 2-D feature matrix.
    Returns:
        torch.Tensor: distance matrix.
    """
    features = F.normalize(features, p=2, dim=1)
    others = F.normalize(others, p=2, dim=1)
    dist_m = 1 - torch.mm(features, others.t())
    return dist_m.cpu().numpy()


def build_dist(feat_1: torch.Tensor, feat_2: torch.Tensor, metric: str = "euclidean", **kwargs) -> np.ndarray:
    r"""Compute distance between two feature embeddings.
    Args:
        feat_1 (torch.Tensor): 2-D feature with batch dimension.
        feat_2 (torch.Tensor): 2-D feature with batch dimension.
        metric:
    Returns:
        numpy.ndarray: distance matrix.
    """
    assert metric in ["cosine", "euclidean", "jaccard"], "Expected metrics are cosine, euclidean and jaccard, " \
                                                         "but got {}".format(metric)

    if metric == "euclidean":
        return compute_euclidean_distance(feat_1, feat_2)

    elif metric == "cosine":
        return compute_cosine_distance(feat_1, feat_2)



class R1_mAP_eval():
    def __init__(self, max_rank=100, metric="cosine", reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.max_rank = max_rank
        self.metric = metric
        self.reranking=reranking

    def reset(self):
        self.query_feats = []
        self.query_pids = []
        self.query_camids = []

        self.gallery_feats = []
        self.gallery_pids = []
        self.gallery_camids = []

    def query_update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.query_feats.append(feat)
        self.query_pids.append(pid)  # 转为 torch 张量
        self.query_camids.append(camid)  # 转为 torch 张量

    def gallery_update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.gallery_feats.append(feat)
        self.gallery_pids.append(pid)  # 转为 torch 张量
        self.gallery_camids.append(camid)  # 转为 torch 张量

    def compute(self):  # called after each epoch
        # query
        qf = torch.cat(self.query_feats, dim=0)
        q_pids = torch.cat(self.query_pids, dim=0).cpu().numpy()
        q_camids = torch.cat(self.query_camids, dim=0).cpu().numpy()

        # gallery
        gf = torch.cat(self.gallery_feats, dim=0)
        g_pids = torch.cat(self.gallery_pids, dim=0).cpu().numpy()
        g_camids = torch.cat(self.gallery_camids, dim=0).cpu().numpy()

        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            k1 = cfg.TEST.RE_RANKING_PARAMETER[0]
            k2 = cfg.TEST.RE_RANKING_PARAMETER[1]
            lambda_value = cfg.TEST.RE_RANKING_PARAMETER[2]
            distmat = re_ranking(qf, gf, k1=k1, k2=k2, lambda_value=lambda_value)

        #
        else:
            print('=> Computing DistMat with cosine similarity')
            distmat = build_dist(qf.float().cpu(), gf.float().cpu(), self.metric)


        cmc, all_AP, all_INP = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids)
        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)

        return cmc, mAP, mINP