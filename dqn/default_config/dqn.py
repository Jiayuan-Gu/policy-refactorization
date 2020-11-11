from yacs.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Config definition
# ---------------------------------------------------------------------------- #
_C = CN(new_allowed=True)
cfg = _C


# ---------------------------------------------------------------------------- #
# Env
# ---------------------------------------------------------------------------- #
_C.env = ''

# ---------------------------------------------------------------------------- #
# Network
# ---------------------------------------------------------------------------- #
_C.network = CN(new_allowed=True)


# ---------------------------------------------------------------------------- #
# RL
# ---------------------------------------------------------------------------- #
_C.RL = CN()
_C.RL.discount = 0.99
_C.RL.max_steps = int(2e7)
_C.RL.state_normalizer = 'ImageNormalizer'
_C.RL.reward_normalizer = 'RescaleNormalizer'
_C.RL.gradient_clip = 10

# ---------------------------------------------------------------------------- #
# DQN
# ---------------------------------------------------------------------------- #
_C.DQN = CN()
_C.DQN.replay = CN()
_C.DQN.replay.is_async = True
_C.DQN.replay.memory_size = int(1e6)
_C.DQN.replay.batch_size = 32
_C.DQN.e_greedy = (1.0, 0.01, int(1e6))
_C.DQN.target_network_update_freq = 10000
_C.DQN.exploration_steps = 50000
_C.DQN.sgd_update_frequency = 4
_C.DQN.double_q = False
_C.DQN.n_batch_per_update = 1
_C.DQN.async_actor = True

# ---------------------------------------------------------------------------- #
# Optimizer
# ---------------------------------------------------------------------------- #
_C.optimizer = CN()

# Basic parameters of the optimizer
# Note that the learning rate should be changed according to batch size
_C.optimizer.lr = 1e-4

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
_C.train = CN()

_C.train.save_interval = 100000
_C.train.log_interval = int(1e3)
_C.train.n_checkpoints_to_keep = 10

# ---------------------------------------------------------------------------- #
# Specific eval options
# ---------------------------------------------------------------------------- #
_C.eval = CN()

_C.eval.interval = 30000
_C.eval.n_episodes = 10
_C.eval.is_async = True
_C.eval.env_subprocess = True
_C.eval.parallel = True
_C.eval.eval_only = False


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# if set to @, the filename of config will be used by default
_C.OUTPUT_DIR = '@'
_C.run_name = ''
_C.load_ckpt = ''
_C.load_conv_from_ae = ''


# ---------------------------------------------------------------------------- #
# Other tmp options
# ---------------------------------------------------------------------------- #
_C.other = CN(new_allowed=True)

