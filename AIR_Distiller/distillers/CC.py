import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

class CC(Distiller):
    """Correlation Congruence for Knowledge Distillation, ICCV 2019.
    The authors nicely shared the code with me. I restructured their code to be 
    compatible with my running framework. Credits go to the original author"""

    def __init__(self, student, teacher, cfg):
        super(CC, self).__init__(student, teacher, cfg)

        self.normalize = cfg.CC.NORMALIZE
        self.kd_loss_weight = cfg.CC.KD_WEIGHT
        

    def forward_train(self, image, kd_student_image, kd_teacher_image, target, kd_target, **kwargs):

        logits_student, feature_student = self.student(image)
        ce_loss = self.ce_loss_weight * self.ce_loss(logits_student, target)
        triplet_loss = self.tri_loss_weight * self.triplet_loss(feature_student["pooled_feat"], target)

        _, kd_feature_student = self.student(kd_student_image)
        _, kd_feature_teacher = self.teacher(kd_teacher_image)

        if self.normalize:
            normalize_fs = F.normalize(kd_feature_student["retrieval_feat"], p=2, dim=1)
            normalize_ft = F.normalize(kd_feature_teacher["retrieval_feat"], p=2, dim=1)
            delta = torch.abs(normalize_fs - normalize_ft)
        else:
            delta = torch.abs(kd_feature_student["retrieval_feat"] - kd_feature_teacher["retrieval_feat"])
         
        kd_loss = self.kd_loss_weight * torch.mean((delta[:-1] * delta[1:]).sum(1))

        losses_dict = {
            "loss_ce": ce_loss,
            "loss_triplet": triplet_loss,
            "loss_kd": kd_loss,
        }
        return logits_student, losses_dict
