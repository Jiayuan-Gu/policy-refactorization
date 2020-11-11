import argparse, json
import pprint
import sys, os
import os.path as osp
import socket
import warnings

# Assume that the script is run at the root directory
_ROOT_DIR = osp.abspath(osp.dirname(__file__) + '/..')
sys.path.insert(0, _ROOT_DIR)

import numpy as np
from rl_libarary.utils.torch_utils import set_one_thread, select_device, random_seed
from rl_libarary.utils.misc import run_steps
from rl_libarary.utils.config import generate_config
from rl_libarary.utils.logger import setup_logger
from rl_libarary.agent.DQN_agent import DQNAgent

def parse_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args

def train_dqn(config):
    run_steps(DQNAgent(config))

def main():
    args = parse_args()
    cfg = generate_config(args)  
    # print(cfg)

    logger = setup_logger('RL', cfg.final_output_dir)
    logger.info('Server: {:s}'.format(socket.gethostname()))
    # logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    # from common.utils.misc import collect_env_info
    # logger.info('Collecting env info (might take some time)\n' + collect_env_info())

    logger.info('Loaded configuration file {:s}'.format(osp.abspath(args.config_file)))
    logger.info('Running with config:\n{}'.format(cfg))
    # import pdb; pdb.set_trace()

    set_one_thread()
    random_seed()
    # select_device(-1)
    select_device(0)

    train_dqn(cfg)

if __name__ == '__main__':
    main()