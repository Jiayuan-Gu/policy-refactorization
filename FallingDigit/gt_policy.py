# -*- coding: utf-8 -*-
import numpy as np
import cv2
import time
import gym

def gt_policy(info):
    agent_y_pos = info['agent_pos'][1]
    target_digit_label = info['target_digit_labels'][0]
    opponents_y_pos = info['opponents_y_pos']
    opponents_label = info['opponents_label']
    for y_pos, label in zip(opponents_y_pos, opponents_label):
        if label == target_digit_label:
            if agent_y_pos < y_pos:
                return 2
            elif agent_y_pos > y_pos:
                return 0
            else:
                return 1
    print('Error!')


if __name__ == '__main__':
    env = gym.make('FallingDigitBlack_3-v1')

    n_ep = 1000
    ep_r_list = []
    for _ in range(n_ep):
        s = env.reset()
        info = env._get_info()
        done = False
        ep_r = 0.0
        while not done:
            action = gt_policy(info)
            s, r, done, info = env.step(action)
            ep_r += r
        ep_r_list.append(ep_r)
    
    print('mean: {:.2f}, std: {:.2f}'.format( np.mean(ep_r_list), np.std(ep_r_list) ))