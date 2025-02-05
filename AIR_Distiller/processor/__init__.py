from .trainer import BaseTrainer, KDTrainer #, CRDTrainer, 

trainer_dict = {
    "vanilla": BaseTrainer,
    "kd": KDTrainer
}
