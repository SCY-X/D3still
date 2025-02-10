import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from .triplet_loss import TripletLoss
from ptflops import get_model_complexity_info

class Distiller(nn.Module):
    def __init__(self, student, teacher, cfg):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

        for p in self.teacher.parameters():
            p.requires_grad = False
        
        logger = logging.getLogger("Asymmetric_Image_Retrieval.train")
        if cfg.EXPERIMENT.TRIPLET_METHOD == "batch_soft":
            self.triplet_loss = TripletLoss(mining_method='batch_soft')
            logger.info("using soft margin triplet loss for training, mining_method: batch_soft")

        else:
            self.triplet_loss = TripletLoss(cfg.EXPERIMENT.TRIPLET_MARGIN, mining_method=cfg.EXPERIMENT.TRIPLET_METHOD)
            logger.info("using Triplet Loss for training with margin:{}, mining_method:{}".format(cfg.EXPERIMENT.TRIPLET_MARGIN, cfg.EXPERIMENT.TRIPLET_METHOD))
        
        if cfg.EXPERIMENT.IF_LABELSMOOTH == 'on':
            self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
            logger.info("Cross entropy with label smoothing")
        else:
            self.ce_loss = nn.CrossEntropyLoss()
            logger.info("Cross entropy")

        self.ce_loss_weight = cfg.EXPERIMENT.CE_LOSS_WEIGHT

        self.tri_loss_weight = cfg.EXPERIMENT.TRIPLET_LOSS_WEIGHT

    def train(self, mode=True):
        # teacher as eval mode by default
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.teacher.eval()
        return self

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        return [(k, v) for k, v in self.student.named_parameters()]

    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0
    
    def get_base_parameters(self):
        num_p = 0
        for name, p in self.student.named_parameters():
            if "classifier" not in name:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
                num_p += p.numel()
        return num_p / 1e6
    
    def get_base_flops(self, resolution):
        self.student.eval()
        input_height, input_width = resolution[0], resolution[1]
       
        image_size = (3, input_height, input_width)
        
        macs, _ = get_model_complexity_info(self.student, image_size, as_strings=True,
                                            print_per_layer_stat=False, verbose=True)
      
        # if macs is None:
        #     print(f"FLOPs calculation failed: {None}, unpacking model...")
        #     # 解封模型后重新计算
        #     unpacked_model = self.unpack_model(self.student)
        #     print(unpacked_model)
            
        #     macs, _ = get_model_complexity_info(nn.Sequential(*unpacked_model), image_size, as_strings=True,
        #                                         print_per_layer_stat=False, verbose=True)
        # print(macs)
        # exit()
        return macs
    
    # def unpack_model(self, model):
    #     """
    #     递归展开模型的所有子模块，但保留原始模型结构。
    #     """
    #     layers = []
    #     for name, module in model.named_children():
    #         print(f"Layer: {name}, Module: {module}")
    #         # 如果模块还有子模块，则继续递归处理
    #         if len(list(module.children())) > 0:
    #             layers.extend(self.unpack_model(module))
    #         else:
    #             layers.append(module)
            
    #     return layers

    def forward_train(self, **kwargs):
        # training function for the distillation method
        raise NotImplementedError()

    def forward_query(self, image):
        _, feature_student = self.student(image)

        return feature_student["retrieval_feat"]
    
    def forward_gallery(self, image):
        _, feature_teacher = self.teacher(image)

        return feature_teacher["retrieval_feat"]

    def forward(self, **kwargs):
        return self.forward_train(**kwargs)


class Vanilla(nn.Module):
    def __init__(self, student, cfg):
        super(Vanilla, self).__init__()
        self.student = student

        logger = logging.getLogger("Asymmetric_Image_Retrieval.train")
        if cfg.EXPERIMENT.TRIPLET_METHOD == "batch_soft":
            self.triplet_loss = TripletLoss(mining_method='batch_soft')
            logger.info("using soft margin triplet loss for training, mining_method: batch_soft")

        else:
            self.triplet_loss = TripletLoss(cfg.EXPERIMENT.TRIPLET_MARGIN, mining_method=cfg.EXPERIMENT.TRIPLET_METHOD)
            logger.info("using Triplet Loss for training with margin:{}, mining_method:{}".format(cfg.EXPERIMENT.TRIPLET_MARGIN, cfg.EXPERIMENT.TRIPLET_METHOD))
        
        if cfg.EXPERIMENT.IF_LABELSMOOTH == 'on':
            self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
            logger.info("Cross entropy with label smoothing")
        else:
            self.ce_loss = nn.CrossEntropyLoss()
            logger.info("Cross entropy")

        self.ce_loss_weight = cfg.EXPERIMENT.CE_LOSS_WEIGHT

        self.tri_loss_weight = cfg.EXPERIMENT.TRIPLET_LOSS_WEIGHT 


    def get_learnable_parameters(self):
        return [(k, v) for k, v in self.student.named_parameters()]

    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0
    
    def get_base_parameters(self):
        num_p = 0
        for name, p in self.student.named_parameters():
            if "classifier" not in name:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
                num_p += p.numel()
        return num_p / 1e6
    
    def get_base_flops(self, resolution):
        self.student.eval()
        input_height, input_width = resolution[0], resolution[1]
        image_size = (3, input_height, input_width)
        macs, _ = get_model_complexity_info(self.student, image_size, as_strings=True,
                                            print_per_layer_stat=False, verbose=True)
        return macs

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        ce_loss = self.ce_loss_weight * self.ce_loss(logits_student, target)
        triplet_loss = self.tri_loss_weight * self.triplet_loss(feature_student["pooled_feat"], target)
        losses_dict = {
            "loss_ce": ce_loss,
            "loss_triplet": triplet_loss,
            "loss_kd": torch.tensor([0.0]).cuda(),
        }
        return logits_student, losses_dict

    def forward_query(self, image):
        _, feature_student = self.student(image)

        return feature_student["retrieval_feat"]
    
    def forward_gallery(self, image):

        _, feature_student = self.student(image)

        return feature_student["retrieval_feat"]
    
    def forward(self, **kwargs):
        return self.forward_train(**kwargs)

   
