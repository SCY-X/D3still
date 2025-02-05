import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller


class FitNet(Distiller):
    """FitNets: Hints for Thin Deep Nets"""

    def __init__(self, student, teacher, cfg):
        super(FitNet, self).__init__(student, teacher, cfg)
      
        self.kd_loss = nn.MSELoss()
        self.kd_loss_weight = cfg.FITNET.KD_WEIGHT
         

    def forward_train(self, image, kd_student_image, kd_teacher_image, target, kd_target, **kwargs):

        logits_student, feature_student = self.student(image)
        ce_loss = self.ce_loss_weight * self.ce_loss(logits_student, target)
        triplet_loss = self.tri_loss_weight * self.triplet_loss(feature_student["pooled_feat"], target)

        _, kd_feature_student = self.student(kd_student_image)
        _, kd_feature_teacher = self.teacher(kd_teacher_image)

        kd_loss = self.kd_loss_weight * self.kd_loss(kd_feature_student["retrieval_feat"], kd_feature_teacher["retrieval_feat"])

        losses_dict = {
            "loss_ce": ce_loss,
            "loss_triplet": triplet_loss,
            "loss_kd": kd_loss,
        }
        return logits_student, losses_dict