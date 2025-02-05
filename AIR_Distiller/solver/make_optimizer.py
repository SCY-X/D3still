import torch
import logging


def make_optimizer(cfg, distiller):
    logger = logging.getLogger("reid_baseline.train")
    params = []
    
    parameter_source = distiller.module if torch.cuda.device_count() > 1 else distiller 
    for key, value in parameter_source.get_learnable_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key:
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.FC_LR_TIMES
                logger.info('Using {} times learning rate for fc'.format(cfg.SOLVER.FC_LR_TIMES))

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
        
    return optimizer
