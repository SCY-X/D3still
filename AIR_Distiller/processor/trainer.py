import os
import time
import logging
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from utils.meter import AverageMeter
from solver import make_optimizer, WarmupMultiStepLR, WarmupCosineAnnealingLR
from .inferencer import inference

class BaseTrainer(object):
    def __init__(self, cfg, distiller, student_train_loader, distillation_loader, query_loader, gallery_loader):
        self.cfg = cfg
        self.distiller = distiller
        self.student_train_loader = student_train_loader
        self.distillation_loader = distillation_loader

        self.query_loader = query_loader
        self.gallery_loader = gallery_loader

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cuda_amp = cfg.EXPERIMENT.CUDA_AMP
        self.checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        self.max_epochs = cfg.SOLVER.MAX_EPOCHS

        self.scaler = GradScaler() if self.cuda_amp else None

        self.logger = logging.getLogger("Asymmetric_Image_Retrieval.train")

        self.optimizer = make_optimizer(cfg, distiller)
    
        if cfg.SOLVER.LR_DECAY_TYPE == 'WarmupMultiStepLR':
            self.scheduler = WarmupMultiStepLR(self.optimizer, cfg.SOLVER.LR_DECAY_STEPS, cfg.SOLVER.LR_DECAY_GAMMA,
                                    cfg.SOLVER.LR_WARMUP_FACTOR, cfg.SOLVER.LR_WARMUP_EPOCHS, cfg.SOLVER.LR_WARMUP_METHOD)
            self.logger.info("use WarmupMultiStepLR, delay_step:{}".format(cfg.SOLVER.STEPS))

        elif cfg.SOLVER.LR_DECAY_TYPE == 'WarmupCosineAnnealingLR':
            self.scheduler = WarmupCosineAnnealingLR(self.optimizer, cfg.SOLVER.MAX_EPOCHS, cfg.SOLVER.LR_DECAY_STEPS[0], cfg.SOLVER.LR_DECAY_ETA_MIN_LR,
                                        cfg.SOLVER.LR_WARMUP_FACTOR, cfg.SOLVER.LR_WARMUP_EPOCHS, cfg.SOLVER.LR_WARMUP_METHOD)
            self.logger.info("use WarmupCosineAnnealingLR, delay_step:{}".format(cfg.SOLVER.LR_DECAY_STEPS[0]))

        self.logger.info("Trainer initialized.")

        # Meters for tracking metrics
        self.loss_meter = AverageMeter()
        self.ce_meter = AverageMeter()
        self.tri_meter = AverageMeter()
        self.kd_meter = AverageMeter()
        self.acc_meter = AverageMeter()

    def reset_meters(self):
        self.loss_meter.reset()
        self.ce_meter.reset()
        self.tri_meter.reset()
        self.kd_meter.reset()
        self.acc_meter.reset()

    def save_checkpoint(self, epoch):
        # save_path = os.path.join(self.cfg.OUTPUT_DIR.ROOT_PATH, self.cfg.OUTPUT_DIR.EXPERIMENT_NAME, "student_{}.pth".format(epoch))

        # model_state = self.distiller.module.student.state_dict() if torch.cuda.device_count() > 1 else self.distiller.student.state_dict()
        # torch.save(model_state, save_path)

        distillaer_save_path = os.path.join(self.cfg.OUTPUT_DIR.ROOT_PATH, self.cfg.OUTPUT_DIR.EXPERIMENT_NAME, f"{self.cfg.DISTILLER.TYPE}_{epoch}.pth")

        model_state = self.distiller.module.state_dict() if torch.cuda.device_count() > 1 else self.distiller.state_dict()
        torch.save(model_state, distillaer_save_path)

    def train_epoch(self, epoch):
        self.reset_meters()
        self.distiller.train()

        start_time = time.time()
        for n_iter, (img, target) in enumerate(self.student_train_loader):
            self.optimizer.zero_grad()

            img, target = img.to(self.device), target.to(self.device)

            if self.cuda_amp:
                with autocast(device_type="cuda"):
                    preds, losses_dict = self.distiller(image=img, target=target)

                loss = sum([l.mean() for l in losses_dict.values()])

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                preds, losses_dict = self.distiller(image=img, target=target)
                loss = sum([l.mean() for l in losses_dict.values()])
                loss.backward()
                self.optimizer.step()

            # Update metrics
            batch_size = img.size(0)
            acc = (preds.max(1)[1] == target).float().mean().item()
            self.loss_meter.update(loss, batch_size)
            self.ce_meter.update(losses_dict["loss_ce"].mean().cpu().detach().numpy(), batch_size)
            self.tri_meter.update(losses_dict["loss_triplet"].mean().cpu().detach().numpy(), batch_size)
            self.kd_meter.update(losses_dict["loss_kd"].mean().cpu().detach().numpy(), batch_size)
            self.acc_meter.update(acc, 1)

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
     
        self.logger.info(
            f"Epoch[{epoch}] Loss: {self.loss_meter.avg:.3f}, CE: {self.ce_meter.avg:.3f}, "
            f"TRI: {self.tri_meter.avg:.3f}, KD: {self.kd_meter.avg:.3f}, Acc: {self.acc_meter.avg:.3f}, "
            f"Base LR: {self.scheduler.get_last_lr()[0]:.2e}, Time/Batch: {time_per_batch:.3f}s"
        )

    def train(self):
        self.logger.info("Starting training...")
        for epoch in range(1, self.max_epochs + 1):
            self.train_epoch(epoch)
            self.scheduler.step()
            if epoch % self.checkpoint_period == 0:
                self.save_checkpoint(epoch)
                inference(self.cfg, self.distiller, self.query_loader, self.gallery_loader)
        self.logger.info("Training completed.")


     


class KDTrainer(BaseTrainer):
    def save_checkpoint(self, epoch):
        distillaer_save_path = os.path.join(self.cfg.OUTPUT_DIR.ROOT_PATH, self.cfg.OUTPUT_DIR.EXPERIMENT_NAME, f"{self.cfg.DISTILLER.TYPE}_{epoch}.pth")

        model_state = self.distiller.module.state_dict() if torch.cuda.device_count() > 1 else self.distiller.state_dict()
        torch.save(model_state, distillaer_save_path)

        student_model = {}
        for name, parameter in model_state.items():
            name_list = name.split(".")
            if "student" in name_list[0]:
                student_model[name] = parameter

        student_save_path = os.path.join(self.cfg.OUTPUT_DIR.ROOT_PATH, self.cfg.OUTPUT_DIR.EXPERIMENT_NAME, "student_{}.pth".format(epoch))
        torch.save(model_state, student_save_path)

    def train_epoch(self, epoch):
        self.reset_meters()
        self.distiller.train()

        start_time = time.time()

        for n_iter, ((img, target), (kd_student_img, kd_teacher_img, kd_target)) in enumerate(zip(self.student_train_loader, self.distillation_loader)):
            self.optimizer.zero_grad()

            img, target = img.to(self.device), target.to(self.device)
            kd_student_img, kd_teacher_img, kd_target = kd_student_img.to(self.device), kd_teacher_img.to(self.device), kd_target.to(self.device)

            if self.cuda_amp:
                with autocast(device_type="cuda"):
                    preds, losses_dict = self.distiller(image=img, kd_student_image=kd_student_img, kd_teacher_image=kd_teacher_img, target=target, kd_target=kd_target)

                loss = sum([l.mean() for l in losses_dict.values()])

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                preds, losses_dict = self.distiller(image=img, kd_student_image=kd_student_img, kd_teacher_image=kd_teacher_img, target=target, kd_target=kd_target)
                loss = sum([l.mean() for l in losses_dict.values()])
                loss.backward()
                self.optimizer.step()

            # Update metrics
            batch_size = img.size(0)
            acc = (preds.max(1)[1] == target).float().mean().item()
            self.loss_meter.update(loss, batch_size)
            self.ce_meter.update(losses_dict["loss_ce"].mean().cpu().detach().numpy(), batch_size)
            self.tri_meter.update(losses_dict["loss_triplet"].mean().cpu().detach().numpy(), batch_size)
            self.kd_meter.update(losses_dict["loss_kd"].mean().cpu().detach().numpy(), batch_size)
            self.acc_meter.update(acc, 1)

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
     
        self.logger.info(
            f"Epoch[{epoch}] Loss: {self.loss_meter.avg:.3f}, CE: {self.ce_meter.avg:.3f}, "
            f"TRI: {self.tri_meter.avg:.3f}, KD: {self.kd_meter.avg:.3f}, Acc: {self.acc_meter.avg:.3f}, "
            f"Base LR: {self.scheduler.get_last_lr()[0]:.2e}, Time/Batch: {time_per_batch:.3f}s"
        )

   


     







