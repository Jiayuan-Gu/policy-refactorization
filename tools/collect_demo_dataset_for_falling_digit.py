from __future__ import division
import os
import os.path as osp
import argparse
import pickle
from tqdm import tqdm

import numpy as np

import random
# import torch
import cv2

import sys
_ROOT_DIR = osp.abspath(osp.dirname(__file__) + '/..')
sys.path.insert(0, _ROOT_DIR)


from rl_libarary.utils.config import generate_eval_config
from rl_libarary.agent.DQN_agent import DQNAgent
from rl_libarary.component.envs import Task

class EnvDataProvider(object):
    def __init__(self, env_name, ckpt, config_file, args):
        self.args = args
        self.config = generate_eval_config(config_file, env_name)
        self.agent = DQNAgent(self.config)
        self.agent.load(ckpt)
        self.agent.evaluator.network.load_state_dict(self.agent.network.state_dict())
        self.env = self.agent.evaluator.task.env.envs[0].env # this env only transpose image

    def get_episode(self, level_idx):
        ep_data = []
        s = self.env.reset(chosen_level_idx=level_idx) # must passed by kwargs
        tot_rewards = 0
        done = False
        while not done:
            original_img = self.env.render(mode='rgb_array') # (128, 128, 3)

            q = self.agent.evaluator.eval_q([s])

            ep_data.append({ 'original_image': original_img, # (128, 128, 3) of uint8
                            'q': q, # (3, ) of float32
                            })

            if self.args.vis:
                cv2.imshow('game', original_img[:, :, ::-1]) # convert RGB to BGR
                cv2.waitKey(1) 
                aaaa = input()

            if done:
                # print('ep_r:', tot_rewards)
                break
            s, r, done, _ = self.env.step(np.argmax(q))
            tot_rewards += r

        if tot_rewards >= self.args.reward_th:
            return ep_data
        else:
            return None

    def close(self):
        self.agent.close()


def parse_args():
    parser = argparse.ArgumentParser(description='abc')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
        required=True,
    )
    parser.add_argument('--output-dir', default=osp.join(osp.dirname(__file__), '../data'),
                        type=str, help='output directory')
    parser.add_argument('--vis', action='store_true', help='whether to visualize')
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--comment', type=str)
    parser.add_argument('--reward-th', default=2.99, type=float)
    parser.add_argument('--start_level', default=0, type=int)
    parser.add_argument('--num_levels', default=3000, type=int)

    args = parser.parse_args()
    return args

def main():
    """
    python tools/collect_demo_dataset_for_falling_digit.py --env FallingDigitCIFAR_3-v1 --cfg configs/falling_digit_rl/dqn_relation_net_eval.yml --ckpt 
    """
    args = parse_args()

    e = EnvDataProvider(env_name=args.env, ckpt=args.ckpt, config_file=args.config_file, args=args)

    results = []
    for i in tqdm(range(args.start_level, args.start_level + args.num_levels)):
        ep_data = e.get_episode(i)
        if ep_data is not None:
            results += ep_data
        
    e.close()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    output_path = osp.join(output_dir, '{:s}_n_{:d}{:s}.pkl'.format(
                                        args.env, len(results),
                                        '_' + args.comment if args.comment else ''))
    print(output_path)

    with open(output_path, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()