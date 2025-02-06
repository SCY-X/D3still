import os
import shutil
import argparse

import torch

import sys
sys.path.append(os.path.abspath("XXXXX/AIR_Distiller"))

from config import cfg
from utils.logger import setup_logger
from dataloader.make_dataloader import DataLoaderFactory
from models import model_dict
from distillers import distiller_dict
from processor.inferencer import inference



if __name__ == "__main__":

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

    output_dir = os.path.join(cfg.OUTPUT_DIR.ROOT_PATH, cfg.OUTPUT_DIR.EXPERIMENT_NAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    shutil.copyfile(args.cfg, os.path.join(output_dir, args.cfg.split("/")[-1]))

    logger = setup_logger("Asymmetric_Image_Retrieval", output_dir, if_train=False)
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

    
    if cfg.DISTILLER.TYPE == "NONE":
        model_student = model_dict[cfg.DISTILLER.STUDENT_NAME](pretrained=True, pretrained_path=cfg.DISTILLER.STUDENT_PRETRAIN_PATH, 
                                                               last_stride=cfg.DISTILLER.STUDENT_LAST_STRIDE, num_classes=num_classes)
        
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student, cfg)
        

    else:
        model_teacher = model_dict[cfg.DISTILLER.TEACHER_NAME](pretrained=False, last_stride=cfg.DISTILLER.TEACHER_LAST_STRIDE, num_classes=num_classes)
        if cfg.DISTILLER.TEACHER_MODEL_PATH is not None:
            model_teacher.load_state_dict(torch.load(cfg.DISTILLER.TEACHER_MODEL_PATH, weights_only=True))

        model_student = model_dict[cfg.DISTILLER.STUDENT_NAME](pretrained=True, pretrained_path=cfg.DISTILLER.STUDENT_PRETRAIN_PATH, 
                                                               last_stride=cfg.DISTILLER.STUDENT_LAST_STRIDE, num_classes=num_classes)
        
        distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )
   
    
    distiller.load_state_dict(torch.load(os.path.join(output_dir, f"{cfg.DISTILLER.TYPE}_{cfg.TEST.WEIGHT}.pth"), weights_only=True))

    if torch.cuda.device_count() > 1:
        print('Using {} GPUs for training'.format(torch.cuda.device_count()))
        distiller = torch.nn.DataParallel(distiller.cuda())
    else:
        distiller = distiller.cuda()

    
    inference(cfg, distiller, query_loader, gallery_loader)

    

   

