from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.EXPERIMENT = CN()
# Using cuda or cpu for training
_C.EXPERIMENT.DEVICE = "cuda"
# Whether to use mixed precision training
_C.EXPERIMENT.CUDA_AMP = True
# ID number of GPU
_C.EXPERIMENT.DEVICE_ID = '0'

# Margin of triplet loss sampler method, option: batch_hard, batch_soft, 'batch_sample'
_C.EXPERIMENT.TRIPLET_METHOD = 'batch_hard'
# Margin of triplet loss
_C.EXPERIMENT.TRIPLET_MARGIN = 0.3
# If train with label smooth, options: 'on', 'off'
_C.EXPERIMENT.IF_LABELSMOOTH = 'on'

# Weight for the Cross-Entropy loss
_C.EXPERIMENT.CE_LOSS_WEIGHT = 1.0
# Weight for the Triplet loss
_C.EXPERIMENT.TRIPLET_LOSS_WEIGHT = 1.0

# # -----------------------------------------------------------------------------
# # distillaer
# # -----------------------------------------------------------------------------

_C.DISTILLER = CN()
_C.DISTILLER.TYPE = "NONE"  # Vanilla as default
_C.DISTILLER.TEACHER_NAME = 'ResNet101'
_C.DISTILLER.TEACHER_LAST_STRIDE = 1
_C.DISTILLER.TEACHER_MODEL_PATH = ""
_C.DISTILLER.STUDENT_NAME = "ResNet18"
_C.DISTILLER.STUDENT_LAST_STRIDE = 1
# 'True' indicates using a model pre-trained on the ImageNet dataset. 
# 'False' means no pre-trained weights will be used, and the model will be trained from scratch.
_C.DISTILLER.STUDENT_PRETRAIN_CHOICE = True
_C.DISTILLER.STUDENT_PRETRAIN_PATH = ""


# # -----------------------------------------------------------------------------
# # Knowledge Distillation
# # -----------------------------------------------------------------------------


# KD CFG
_C.VanillaKD = CN()
_C.VanillaKD.TEMPERATURE = 2.0
_C.VanillaKD.KD_WEIGHT = 1.0


# FITNET CFG
_C.FITNET = CN()
_C.FITNET.KD_WEIGHT = 2.0


# CCKD CFG
_C.CC = CN()
_C.CC.KD_WEIGHT = 1.0
_C.CC.NORMALIZE = True

# RKD CFG
_C.RKD = CN()
_C.RKD.DISTANCE_WEIGHT = 25
_C.RKD.ANGLE_WEIGHT = 50
_C.RKD.KD_WEIGHT = 1.0
_C.RKD.PDIST = CN()
_C.RKD.PDIST.EPSILON = 1e-12
_C.RKD.PDIST.SQUARED = False

# PKT CFG
_C.PKT = CN()
_C.PKT.KD_WEIGHT = 30000.0


# CSD CFG
_C.CSD = CN()
_C.CSD.TOPK = 96
_C.CSD.TEMPERATURE_QUERY = 1
_C.CSD.TEMPERATURE_GALLERY = 0.01
_C.CSD.KD_WEIGHT = 1.0


# ROP CFG
_C.ROP = CN()
_C.ROP.TOPK = 96
_C.ROP.TEMPERATURE = 0.1
_C.ROP.RANK_WEIGHT = 0.2
_C.ROP.KD_WEIGHT = 1.0

# RAML CFG
_C.RAML = CN()
_C.RAML.LAMBDA1 = 0.7482
_C.RAML.LAMBDA2 = 0.6778
_C.RAML.KD_WEIGHT = 1.0


# D3 CFG
_C.D3 = CN()
_C.D3.TOPK = 10
_C.D3.ALPHA = 50.0
_C.D3.BETA = 2.0
_C.D3.GAMMA = 1.0
_C.D3.KD_WEIGHT = 1.0

# UGD CFG
_C.UGD = CN()
_C.UGD.DISTILLATION_LAYER = 3
_C.UGD.ALPHA = 2.0
_C.UGD.BETA = 2.0
_C.UGD.KD_WEIGHT = 1.0

# # -----------------------------------------------------------------------------
# # INPUT
# # -----------------------------------------------------------------------------

_C.INPUT = CN()
# Size of the image during training
_C.INPUT.STUDENT_SIZE_TRAIN = [64, 64]
# Size of the image during test
_C.INPUT.STUDENT_SIZE_TEST = [64, 64]
# Value of padding size of the student
_C.INPUT.STUDENT_PADDING = 2 #32:1, 64:2, 256:8 384:12


# Size of the image during training
_C.INPUT.TEACHER_SIZE_TRAIN = [256, 256]
# Size of the student image during test
_C.INPUT.TEACHER_SIZE_TEST = [256, 256]
# Value of padding size of the teacher
_C.INPUT.TEACHER_PADDING = 8 #32:1, 64:2, 256:8 384:12

# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('SOP')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('/home/data1/xieyi/data')

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'triplet'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 6

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.TRAINER = "vanilla"
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 96
_C.SOLVER.IMS_DISTILLATION_PER_BATCH = 256

# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = 'SGD'
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 120
# Base learning rate
_C.SOLVER.BASE_LR = 0.01
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = True
#the time learning rate of fc layer
_C.SOLVER.FC_LR_TIMES = 1.0
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

#lr_scheduler
#  warm up epochs
_C.SOLVER.LR_WARMUP_EPOCHS = 10
# warm up factor
_C.SOLVER.LR_WARMUP_FACTOR = 0.1
# method of warm up, option: 'constant','linear'
_C.SOLVER.LR_WARMUP_METHOD = "linear"

#lr_scheduler method, option WarmupMultiStepLR, WarmupCosineAnnealingLR
_C.SOLVER.LR_DECAY_TYPE = 'WarmupCosineAnnealingLR'
# decay rate of learning rate
_C.SOLVER.LR_DECAY_GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.LR_DECAY_STEPS = [40, 70]
#The cosine annealing learning rate drops to the minimum learning rate
_C.SOLVER.LR_DECAY_ETA_MIN_LR = 1e-7


# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = _C.SOLVER.MAX_EPOCHS
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 270
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = _C.SOLVER.MAX_EPOCHS

# set random seed
_C.SOLVER.SEED = 2024

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 512
# Whether using fliped feature for testing, option: 'on', 'off'
_C.TEST.FLIP_FEATS = 'off'
# Path to trained model
_C.TEST.WEIGHT = _C.SOLVER.MAX_EPOCHS
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.TEST_METRIC = 'cosine'
_C.TEST.RE_RANKING = False
# K1, K2, LAMBDA
_C.TEST.RE_RANKING_PARAMETER = [60, 10, 0.3]

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = CN()
_C.OUTPUT_DIR.ROOT_PATH = "./log"
_C.OUTPUT_DIR.EXPERIMENT_NAME = ""

