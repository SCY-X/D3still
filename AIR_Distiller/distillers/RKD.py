import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller



def _pdist(e, squared, eps):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


def rkd_loss(f_s, f_t, squared=False, eps=1e-12, distance_weight=25, angle_weight=50):
    stu = f_s.view(f_s.shape[0], -1)
    tea = f_t.view(f_t.shape[0], -1)

    # RKD distance loss
    with torch.no_grad():
        t_d = _pdist(tea, squared, eps)
        mean_td = t_d[t_d > 0].mean()
        t_d = t_d / mean_td

    d = _pdist(stu, squared, eps)
    mean_d = d[d > 0].mean()
    d = d / mean_d

    loss_d = F.smooth_l1_loss(d, t_d)

    # RKD Angle loss
    with torch.no_grad():
        td = tea.unsqueeze(0) - tea.unsqueeze(1)
        norm_td = F.normalize(td, p=2, dim=2)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

    sd = stu.unsqueeze(0) - stu.unsqueeze(1)
    norm_sd = F.normalize(sd, p=2, dim=2)
    s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

    loss_a = F.smooth_l1_loss(s_angle, t_angle)

    loss = distance_weight * loss_d + angle_weight * loss_a
    return loss


class RKD(Distiller):
    """Relational Knowledge Disitllation, CVPR2019"""

    def __init__(self, student, teacher, cfg):
        super(RKD, self).__init__(student, teacher, cfg)
      
        self.distance_weight = cfg.RKD.DISTANCE_WEIGHT
        self.angle_weight = cfg.RKD.ANGLE_WEIGHT
        self.kd_loss_weight = cfg.RKD.KD_WEIGHT
        self.eps = cfg.RKD.PDIST.EPSILON
        self.squared = cfg.RKD.PDIST.SQUARED
         

    def forward_train(self, image, kd_student_image, kd_teacher_image, target, kd_target, **kwargs):

        logits_student, feature_student = self.student(image)
        ce_loss = self.ce_loss_weight * self.ce_loss(logits_student, target)
        triplet_loss = self.tri_loss_weight * self.triplet_loss(feature_student["pooled_feat"], target)

        _, kd_feature_student = self.student(kd_student_image)
        _, kd_feature_teacher = self.teacher(kd_teacher_image)

        kd_loss = self.kd_loss_weight * rkd_loss(
            kd_feature_student["retrieval_feat"],
            kd_feature_teacher["retrieval_feat"],
            self.squared,
            self.eps,
            self.distance_weight,
            self.angle_weight)

        losses_dict = {
            "loss_ce": ce_loss,
            "loss_triplet": triplet_loss,
            "loss_kd": kd_loss,
        }
        return logits_student, losses_dict

    # def forward_query(self, image):
    #     _, feature_student = self.student(image)

    #     return feature_student["retrieval_feat"]
    
    # def forward_gallery(self, image):

    #     _, feature_teacher = self.teacher(image)

    #     return feature_teacher["retrieval_feat"]
    
    # def forward(self, **kwargs):
    #     return self.forward_train(**kwargs)