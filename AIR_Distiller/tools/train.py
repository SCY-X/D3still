import os
import shutil
import random
import numpy as np
import argparse
import torch

from config import cfg
from utils.logger import setup_logger
from models import model_dict
from distillers import distiller_dict
from dataloader.make_dataloader import DataLoaderFactory
from processor import trainer_dict

import sys
sys.path.append(os.path.abspath("XXXXX/AIR_Distiller"))

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main(cfg, student_train_loader, distillation_loader, num_classes, query_loader, gallery_loader):

    if cfg.DISTILLER.TYPE == "NONE":
        model_student = model_dict[cfg.DISTILLER.STUDENT_NAME](pretrained=True, pretrained_path=cfg.DISTILLER.STUDENT_PRETRAIN_PATH, 
                                                               last_stride=cfg.DISTILLER.STUDENT_LAST_STRIDE, num_classes=num_classes)
        
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student, cfg)

    else:
        model_teacher = model_dict[cfg.DISTILLER.TEACHER_NAME](pretrained=False, last_stride=cfg.DISTILLER.TEACHER_LAST_STRIDE, num_classes=num_classes)
        if cfg.DISTILLER.TEACHER_MODEL_PATH is not None:
            model_teacher.load_state_dict({k.replace('student.', ""): v for k, v in torch.load(cfg.DISTILLER.TEACHER_MODEL_PATH, weights_only=True).items()})

        
        model_student = model_dict[cfg.DISTILLER.STUDENT_NAME](pretrained=True, pretrained_path=cfg.DISTILLER.STUDENT_PRETRAIN_PATH, 
                                                               last_stride=cfg.DISTILLER.STUDENT_LAST_STRIDE, num_classes=num_classes)
        
        distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg)

    if torch.cuda.device_count() > 1:
        print('Using {} GPUs for training'.format(torch.cuda.device_count()))
        distiller = torch.nn.DataParallel(distiller.cuda())
    else:
        distiller = distiller.cuda()

    
    if torch.cuda.device_count() > 1:
        base_params = distiller.module.get_base_parameters()
        extra_params = distiller.module.get_extra_parameters()
        base_flops = distiller.module.get_base_flops()
    else:
        base_params = distiller.get_base_parameters()
        extra_params = distiller.get_extra_parameters()
        base_flops = distiller.get_base_flops(cfg.INPUT.STUDENT_SIZE_TRAIN)

    print("The student base parameters: {:.3f} M, Extra parameters of {}: {:.3f} M, and base flops: {:<8}\033[0m".format(
        base_params, cfg.DISTILLER.TYPE, extra_params, base_flops))
    
    output_dir = os.path.join(cfg.OUTPUT_DIR.ROOT_PATH, cfg.OUTPUT_DIR.EXPERIMENT_NAME)
    with open('%s/inference_speed.txt' % output_dir, 'a') as test_file:
        test_file.write('{:<30}  {:.2f} M\n'.format('Base Parameters: ', base_params))
        test_file.write('{:<30}  {:.2f} M\n'.format('Extra Parameters: ', extra_params))
        test_file.write('{:<30}  {:<8} \n'.format('Base FLOPs: ', base_flops))
        test_file.write('-----------------\n')

    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        cfg, distiller, student_train_loader, distillation_loader, query_loader, gallery_loader)
    
    trainer.train()
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Asymmetric Image Retrieval Training")
    parser.add_argument(
        "--cfg", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.cfg != "":
        cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    output_dir = os.path.join(cfg.OUTPUT_DIR.ROOT_PATH, cfg.OUTPUT_DIR.EXPERIMENT_NAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    shutil.copyfile(args.cfg, os.path.join(output_dir, args.cfg.split("/")[-1]))

    logger = setup_logger("Asymmetric_Image_Retrieval", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(output_dir))
    logger.info(args)

    if args.cfg != "":
        logger.info("Loaded configuration file {}".format(args.cfg))
        with open(args.cfg, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.EXPERIMENT.DEVICE_ID
    
    dataloader_factory = DataLoaderFactory(cfg)
    student_train_loader, distillation_loader, query_loader, gallery_loader, num_classes = dataloader_factory.create_dataloaders()

    main(cfg, student_train_loader, distillation_loader, num_classes, query_loader, gallery_loader)
