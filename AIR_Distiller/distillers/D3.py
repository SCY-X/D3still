import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller

def d3_loss(student_features, teacher_features, topk, alpha, beta, gamma): 

    batch_size = student_features.shape[0]

    # Normalize student and teacher features
    student_features = F.normalize(student_features, p=2, dim=1)
    teacher_features = F.normalize(teacher_features, p=2, dim=1)

    # Compute similarity matrices
    teacher_similarity = teacher_features.double().mm(teacher_features.double().t())
    cross_similarity = student_features.double().mm(teacher_features.double().t())

    # Get sorted teacher similarity and corresponding cross similarity
    teacher_topk_values, sorted_indices = torch.sort(teacher_similarity, dim=1, descending=True)
    student_topk_values = torch.gather(cross_similarity, 1, sorted_indices)

    # Feature distillation loss (alpha term)
    fd_loss = alpha * torch.norm(teacher_topk_values[:, 0] - student_topk_values[:, 0], p=2) / batch_size

    # Extract top-k similarities (excluding the highest)
    student_distances = student_topk_values[:, 1:topk]
    teacher_distances = teacher_topk_values[:, 1:topk]
   
    
    # Compute pairwise difference matrices
    student_diff_matrix = student_distances.unsqueeze(1) - student_distances.unsqueeze(2)
    teacher_diff_matrix = teacher_distances.unsqueeze(1) - teacher_distances.unsqueeze(2)

    # Flatten the difference matrices
    student_diff_flat = student_diff_matrix.view(batch_size, -1)
    teacher_diff_flat = teacher_diff_matrix.view(batch_size, -1)
    
    # Avoid division by zero
    teacher_diff_flat[teacher_diff_flat == 0] = 1

    # Compute hard and simple weights
    hard_weights = (student_diff_flat / teacher_diff_flat).detach()
    simple_weights = (student_diff_flat / teacher_diff_flat).detach()

    hard_weights[hard_weights >= 0] = 0
    hard_weights[hard_weights < 0] = 1

    simple_weights[simple_weights <= 0] = 0
    simple_weights[simple_weights > 0] = 1

    # Avoid division by zero in student differences
    student_diff_flat[student_diff_flat == 0] = 1

    # Compute weighted result matrices
    hard_loss_matrix = hard_weights * ((student_diff_flat - teacher_diff_flat) / (0.1 + teacher_diff_flat.abs()))
    simple_loss_matrix = simple_weights * ((student_diff_flat - teacher_diff_flat) / (0.1 + teacher_diff_flat.abs()))

    # Relation KD loss (beta and gamma terms)
    hard_rd_loss = beta * torch.mean(torch.norm(hard_loss_matrix, p=2, dim=1)) / (topk-1)

    simple_rd_loss = gamma * torch.mean(torch.norm(simple_loss_matrix, p=2, dim=1)) / (topk-1)
    
    return fd_loss +  hard_rd_loss + simple_rd_loss



class D3(Distiller):
    """D3still: Decoupled Differential Distillation for Asymmetric Image Retrieval. CVPR2024"""

    def __init__(self, student, teacher, cfg):
        super(D3, self).__init__(student, teacher, cfg)

        self.topk = cfg.D3.TOPK
        self.alpha = cfg.D3.ALPHA
        self.beta = cfg.D3.BETA
        self.gamma = cfg.D3.GAMMA

        self.kd_loss_weight = cfg.D3.KD_WEIGHT

    def forward_train(self, image, kd_student_image, kd_teacher_image, target, kd_target, **kwargs):

        logits_student, feature_student = self.student(image)
        ce_loss = self.ce_loss_weight * self.ce_loss(logits_student, target)
        triplet_loss = self.tri_loss_weight * self.triplet_loss(feature_student["pooled_feat"], target)

        _, kd_feature_student = self.student(kd_student_image)
        _, kd_feature_teacher = self.teacher(kd_teacher_image)

        kd_loss = self.kd_loss_weight * d3_loss(kd_feature_student["retrieval_feat"], 
                                                 kd_feature_teacher["retrieval_feat"],
                                                 self.topk,
                                                 self.alpha,
                                                 self.beta,
                                                 self.gamma)

        losses_dict = {
            "loss_ce": ce_loss,
            "loss_triplet": triplet_loss,
            "loss_kd": kd_loss,
        }
        return logits_student, losses_dict
