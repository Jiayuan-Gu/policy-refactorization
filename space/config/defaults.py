from yacs.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Config definition
# ---------------------------------------------------------------------------- #
_C = CN()
cfg = _C

# ---------------------------------------------------------------------------- #
# Resume
# ---------------------------------------------------------------------------- #
# Automatically resume weights from last checkpoints
_C.AUTO_RESUME = True
# Whether to resume the optimizer and the scheduler
_C.RESUME_STATES = True
# Path of weights to resume
_C.RESUME_PATH = ''
_C.RESUME_STRICT = True

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.TYPE = ''

# If you do not register a model here, then the default arguments will be used.
_C.MODEL.SPACE_v0 = CN(new_allowed=True)

# ---------------------------------------------------------------------------- #
# Prior
# ---------------------------------------------------------------------------- #
_C.PRIOR = CN(new_allowed=True)

# ---------------------------------------------------------------------------- #
# Weight
# ---------------------------------------------------------------------------- #
_C.WEIGHT = CN(new_allowed=True)

# ---------------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------------- #
_C.DATASET = CN()
_C.DATASET.NAME = ''
_C.DATASET.TRAIN = CN(new_allowed=True)
_C.DATASET.VAL = CN(new_allowed=True)

# ---------------------------------------------------------------------------- #
# Optimizer
# ---------------------------------------------------------------------------- #
_C.OPTIMIZER = CN()
_C.OPTIMIZER.TYPE = ''

# Basic parameters of the optimizer
# Note that the learning rate should be changed according to batch size
_C.OPTIMIZER.LR = 1e-3
_C.OPTIMIZER.WEIGHT_DECAY = 0.0
# Maximum norm of gradients. Non-positive for disable
_C.OPTIMIZER.MAX_GRAD_NORM = 0.0

# Specific parameters of optimizers
_C.OPTIMIZER.SGD = CN()
_C.OPTIMIZER.SGD.momentum = 0.9

_C.OPTIMIZER.Adam = CN()
_C.OPTIMIZER.Adam.betas = (0.9, 0.999)

# ---------------------------------------------------------------------------- #
# Scheduler (learning rate schedule)
# ---------------------------------------------------------------------------- #
_C.LR_SCHEDULER = CN()
_C.LR_SCHEDULER.TYPE = ''

# Specific parameters of schedulers
_C.LR_SCHEDULER.StepLR = CN()
_C.LR_SCHEDULER.StepLR.step_size = 0
_C.LR_SCHEDULER.StepLR.gamma = 0.1

_C.LR_SCHEDULER.MultiStepLR = CN()
_C.LR_SCHEDULER.MultiStepLR.milestones = ()
_C.LR_SCHEDULER.MultiStepLR.gamma = 0.1

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

# Batch size
_C.TRAIN.BATCH_SIZE = 1
# Number of workers (dataloader)
_C.TRAIN.NUM_WORKERS = 0
# Period to save checkpoints. 0 for disable
_C.TRAIN.CHECKPOINT_PERIOD = 0
# Period to log training status. 0 for disable
_C.TRAIN.LOG_PERIOD = 0
# Period to summary training status. 0 for disable
_C.TRAIN.SUMMARY_PERIOD = 0
# Max number of checkpoints to keep
_C.TRAIN.MAX_TO_KEEP = 0
# Max number of iteration
_C.TRAIN.MAX_ITER = 1

_C.TRAIN.FROZEN_MODULES = ()
_C.TRAIN.FROZEN_PARAMS = ()

# ---------------------------------------------------------------------------- #
# Specific validation options
# ---------------------------------------------------------------------------- #
_C.VAL = CN()

# Batch size
_C.VAL.BATCH_SIZE = 1
# Number of workers (dataloader)
_C.VAL.NUM_WORKERS = 0
# Period to validate. 0 for disable
_C.VAL.PERIOD = 0
# Period to log validation status. 0 for disable
_C.VAL.LOG_PERIOD = 0
# The metric for best validation performance
_C.VAL.METRIC = ''
_C.VAL.METRIC_ASCEND = True
_C.VAL.EVAL_DETECTION = False

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# if set to @, the filename of config will be used by default
_C.OUTPUT_DIR = '@'

# For reproducibility...but not really because modern fast GPU libraries use
# non-deterministic op implementations
# -1 means use time seed.
_C.RNG_SEED = -1
