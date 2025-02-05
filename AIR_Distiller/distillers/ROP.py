import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller

def sigmoid(x, tau):
    y = 1.0 / (1.0 + torch.exp(-x / tau))
    return y

def rank_order_preservation(student_features, teacher_features, topk, temperature, rank_weight): 
    # Rank Order Preservation

    # Normalize features with L2 normalization
    student_features = F.normalize(student_features, p=2, dim=1)
    teacher_features = F.normalize(teacher_features, p=2, dim=1)

    # Compute similarity matrix for teacher features
    teacher_similarity = teacher_features.double().mm(teacher_features.double().t())
    # Compute cross-similarity matrix between student and teacher features
    cross_similarity = student_features.double().mm(teacher_features.double().t())

    # Get the top-k indices of teacher similarity for retrieval
    teacher_topk_values, teacher_topk_indices = torch.sort(teacher_similarity, dim=1, descending=True)
    topk_gallery_indices = teacher_topk_indices[:, :topk]

    # Gather top-k similarity scores for student predictions
    student_topk_similarity = torch.gather(cross_similarity, 1, topk_gallery_indices)

    # Extract top-k similarity scores from teacher similarity matrix
    teacher_topk_similarity = teacher_topk_values[:, :topk]

    # Compute teacher similarity differences (Indicator matrix \bm{I}_g, Eq. (8))
    teacher_similarity_expanded = teacher_topk_similarity.unsqueeze(1).repeat(1, topk, 1)
    teacher_similarity_diff = teacher_similarity_expanded - teacher_similarity_expanded.permute(0, 2, 1)

    # Pass through the sigmoid to create the indicator matrix
    teacher_indicator_matrix = sigmoid(teacher_similarity_diff, 1e-6)  # Tensor shape: B x topk x topk

    # Compute rank weights (Eq. 9)
    similarity_weights = F.softmax(teacher_topk_similarity * rank_weight, dim=-1)  # Tensor shape: B x topk
    positional_weights = torch.arange(start=1, end=topk + 1, dtype=teacher_features.dtype).unsqueeze(0).cuda()  # Tensor shape: 1 x topk

    rank_weights_matrix = (similarity_weights * positional_weights).unsqueeze(-1)  # Tensor shape: B x topk x 1

    # Compute student similarity differences (Indicator matrix \bm{I}_q, Eq. (10))
    student_similarity_expanded = student_topk_similarity.unsqueeze(1).repeat(1, topk, 1)
    student_similarity_diff = student_similarity_expanded - teacher_similarity_expanded.permute(0, 2, 1)

    # Pass through the sigmoid to create the indicator matrix for students
    student_indicator_matrix = sigmoid(student_similarity_diff, temperature)

    # Compute the rank order preservation loss
    loss = torch.sum(torch.pow(teacher_indicator_matrix - student_indicator_matrix, 2) * rank_weights_matrix, dim=(-1, -2)).mean() / topk

    return loss



class ROP(Distiller):
    """A General Rank Preserving Framework for Asymmetric Image Retrieval. ICLR2023"""

    """
    The original paper adopts an offline approach to extracting gallery features from the entire training set. 
    However, through practical experiments, it was found that directly extracting gallery features for each batch achieves better performance. 
    Therefore, in this project, I chose to use the latter approach. It is important to note that this modification is based on 
    personal experimentation and is not directly related to the original paper. The effectiveness of this approach may vary 
    depending on the specific dataset or experimental settings.
    """

    def __init__(self, student, teacher, cfg):
        super(ROP, self).__init__(student, teacher, cfg)

        self.topk = cfg.ROP.TOPK
        self.temperature = cfg.ROP.TEMPERATURE
        self.rank_weight = cfg.ROP.RANK_WEIGHT
        self.kd_loss_weight = cfg.ROP.KD_WEIGHT


    def forward_train(self, image, kd_student_image, kd_teacher_image, target, kd_target, **kwargs):

        logits_student, feature_student = self.student(image)
        ce_loss = self.ce_loss_weight * self.ce_loss(logits_student, target)
        triplet_loss = self.tri_loss_weight * self.triplet_loss(feature_student["pooled_feat"], target)

        _, kd_feature_student = self.student(kd_student_image)
        _, kd_feature_teacher = self.teacher(kd_teacher_image)

        kd_loss = self.kd_loss_weight * rank_order_preservation(kd_feature_student["retrieval_feat"], 
                                                 kd_feature_teacher["retrieval_feat"], 
                                                 self.topk,
                                                 self.temperature,
                                                 self.rank_weight)

        losses_dict = {
            "loss_ce": ce_loss,
            "loss_triplet": triplet_loss,
            "loss_kd": kd_loss,
        }
        return logits_student, losses_dict
