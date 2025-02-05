import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller

def regression_loss(x1, x2, eps=1e-6):
    dif = 1 - (x1 * x2).sum(-1)
    D = torch.pow(dif + eps, 2)
    return torch.sum(D.mean(-1))

def mse_loss(x1, x2, eps=1e-6):
    dif = x1 - x2
    D = torch.pow(dif + eps, 2)
    return D.fill_diagonal_(0).sum() / (D.shape[0] * (D.shape[1]-1))

class SecondOrderLoss(nn.Module):
    def __init__(self, lam_1=0.7482, lam_2=0.6778, eps=1e-6):
        super(SecondOrderLoss, self).__init__()
        self.eps = eps
        self.lam_1 = lam_1
        self.lam_2 = lam_2
        self.rel = lam_1 or lam_2

    def forward(self, s_vecs, t_vecs):
        s_vecs = F.normalize(s_vecs, p=2, dim=1)
        t_vecs = F.normalize(t_vecs, p=2, dim=1)

        # abs
        loss = regression_loss(s_vecs, t_vecs, eps=self.eps)

        if self.rel:
            t_sim = torch.mm(t_vecs, t_vecs.t())
            # rel_ts
            if self.lam_1:
                s_t_sim = torch.mm(s_vecs, t_vecs.t())
                loss += self.lam_1 * mse_loss(s_t_sim, t_sim)

            # rel_ss
            if self.lam_2:
                s_sim = torch.mm(s_vecs, s_vecs.t())
                loss += self.lam_2 * mse_loss(s_sim, t_sim)

        return loss



class RAML(Distiller):
    """Large-to-Small Image Resolution Asymmetry in Deep Metric Learning. WACV2023"""

    def __init__(self, student, teacher, cfg):
        super(RAML, self).__init__(student, teacher, cfg)

        self.lam_1 = cfg.RAML.LAMBDA1
        self.lam_2 = cfg.RAML.LAMBDA1
        self.kd_loss = SecondOrderLoss(self.lam_1, self.lam_2)
        self.kd_loss_weight = cfg.RAML.KD_WEIGHT

    def forward_train(self, image, kd_student_image, kd_teacher_image, target, kd_target, **kwargs):

        logits_student, feature_student = self.student(image)
        ce_loss = self.ce_loss_weight * self.ce_loss(logits_student, target)
        triplet_loss = self.tri_loss_weight * self.triplet_loss(feature_student["pooled_feat"], target)

        _, kd_feature_student = self.student(kd_student_image)
        _, kd_feature_teacher = self.teacher(kd_teacher_image)

        kd_loss = self.kd_loss_weight * self.kd_loss(kd_feature_student["retrieval_feat"], 
                                                 kd_feature_teacher["retrieval_feat"])

        losses_dict = {
            "loss_ce": ce_loss,
            "loss_triplet": triplet_loss,
            "loss_kd": kd_loss,
        }
        return logits_student, losses_dict
