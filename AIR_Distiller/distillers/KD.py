import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller

def vanillakd_loss(logits_student_in, logits_teacher_in, temperature):
    log_pred_student = F.log_softmax(logits_student_in / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher_in / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

class VanillaKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(VanillaKD, self).__init__(student, teacher, cfg)
      
        self.temperature = cfg.VanillaKD.TEMPERATURE
        self.kd_loss_weight = cfg.VanillaKD.KD_WEIGHT
         

    def forward_train(self, image, kd_student_image, kd_teacher_image, target, kd_target, **kwargs):

        logits_student, feature_student = self.student(image)
        ce_loss = self.ce_loss_weight * self.ce_loss(logits_student, target)
        triplet_loss = self.tri_loss_weight * self.triplet_loss(feature_student["pooled_feat"], target)

        kd_logits_student, kd_feature_student = self.student(kd_student_image)
        kd_logits_teacher, kd_feature_teacher = self.teacher(kd_teacher_image)

        kd_loss = self.kd_loss_weight * vanillakd_loss(kd_logits_student, kd_logits_teacher, self.temperature)

        losses_dict = {
            "loss_ce": ce_loss,
            "loss_triplet": triplet_loss,
            "loss_kd": kd_loss,
        }
        return logits_student, losses_dict
