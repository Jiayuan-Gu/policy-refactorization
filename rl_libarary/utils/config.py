from ..component.envs import Task
from ..utils.misc import has_flag
from yacs.config import CfgNode as CN
import os
import os.path as osp
import warnings
import time

def setup_env_in_config(config):
    config.task = CN(new_allowed=True)
    config.task.full_name = config.env
    tmp_env = Task(config.task.full_name)
    config.task.state_dim = tmp_env.state_dim
    config.task.action_dim = tmp_env.action_dim

def setup_config(config, args):
    # --------- debug
    config.debug = args.debug

    # -------- env related
    setup_env_in_config(config)

    # -------- output_dir
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    run_name = '{:s}_{:s}'.format(timestamp, config.env)
    if config.run_name:
        run_name += ('_' + config.run_name)

    output_dir = config.OUTPUT_DIR if not args.debug else ''
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs', 'outputs'))
        output_dir = osp.join(output_dir, run_name)
        if osp.isdir(output_dir):
            warnings.warn('Output directory exists.')
        os.makedirs(output_dir, exist_ok=True)
    config.final_output_dir = output_dir

def generate_config(args):
    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from dqn.default_config.dqn import cfg
    from common.utils.cfg_utils import purge_cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    setup_config(cfg, args)
    # import pdb; pdb.set_trace()
    cfg.freeze()
    return cfg

def generate_eval_config(config_file, env):
    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from dqn.default_config.dqn import cfg
    from common.utils.cfg_utils import purge_cfg
    cfg.merge_from_file(config_file)
    cfg.env = env
    purge_cfg(cfg)
    setup_env_in_config(cfg)
    cfg.final_output_dir = ''
    cfg.freeze()
    return cfg