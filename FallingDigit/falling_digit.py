import numpy as np
import gym
from gym import spaces
import torch
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from collections import namedtuple
import cv2
import os
from skimage.util import img_as_ubyte
import random

Digit = namedtuple('Digit', ('image', 'label', 'mask'))

class FallingDigitEnv(gym.Env):
    def __init__(self, n_opponents=3, 
                    img_size=(128, 128), 
                    grid_size=14, 
                    max_ep_steps=100, 
                    training_split=True, 
                    bg_type='black',
                    start_level=0,
                    num_levels=0,
                    n_instance_per_digit=None,
                    n_bg_images=None,
                    ):
        self.np_rng = np.random.RandomState()
        self.np_rng_for_level_selection = np.random.RandomState()
        self.n_opponents = n_opponents
        self.img_size = img_size
        self.max_ep_steps = max_ep_steps
        self.start_level = start_level
        self.num_levels = num_levels
        self.observation_space =  spaces.Box(low=0, high=255, shape=(img_size[0], img_size[1], 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(3)

        self.mnist_dataset = MNIST(root='/tmp', train=training_split, download=True)
        if n_instance_per_digit is not None:
            digit_cnt = [0 for _ in range(10)]
            digit_enough_cnt = 0
            sub_indices = []
            for idx in range(len(self.mnist_dataset)):
                k = int(self.mnist_dataset.targets[idx])
                if digit_cnt[k] < n_instance_per_digit:
                    sub_indices.append(idx)
                    digit_cnt[k] += 1
                    if digit_cnt[k] == n_instance_per_digit:
                        digit_enough_cnt +=1
                        if digit_enough_cnt == 10:
                            break           
            self.mnist_dataset.data = self.mnist_dataset.data[sub_indices]
            self.mnist_dataset.targets = self.mnist_dataset.targets[sub_indices]
        self.mnist_dataset_size = len(self.mnist_dataset)

        self.bg_type = bg_type
        if bg_type == 'cifar':
            self.cifar_dataset = CIFAR10(root='/tmp', train=training_split, download=True).data
            if n_bg_images is not None:
                torch.manual_seed(1)
                new_idx = torch.randperm(len(self.cifar_dataset)).numpy()
                self.cifar_dataset = self.cifar_dataset[new_idx] # random shuffle
                self.cifar_dataset = self.cifar_dataset[:n_bg_images]
            self.cifar_dataset_size = len(self.cifar_dataset)

        self.grid_size = grid_size
        self.n_grid = [ int(x/grid_size) for x in img_size]
        self.boundary = [ int((x % grid_size)/2) for x in img_size]

        self.right_hit_reward = 1.0
        self.wrong_hit_reward = -1.0
        self.miss_reward = -1.0

        self.action_to_y_pos_change = [-1, 0, 1]
        self.agent_color = np.array([1.0, 0.0, 0.0])

        self.reset()

    def reset(self, chosen_level_idx=None):
        if chosen_level_idx is not None:
            self.seed(chosen_level_idx)
        elif self.num_levels > 0:
            level_idx = self.np_rng_for_level_selection.randint(low=self.start_level, high=self.start_level + self.num_levels) # must use another random source!!!
            # print('level index:', level_idx)
            self.seed(level_idx)
        self.num_steps_so_far = 0
        self._create_new_opponents()
        self._create_new_agent() # must be called after create opponents
        self._set_bg_img()
        self._draw_static_obj_and_bg()
        return self._draw_screen()

    def _set_bg_img(self):
        if self.bg_type == 'black':
            self.bg_img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        elif self.bg_type == 'white':
            self.bg_img = np.ones((self.img_size[0], self.img_size[1], 3), dtype=np.uint8) * 255
        elif self.bg_type == 'cifar':
            background = self.cifar_dataset[self.np_rng.randint(self.cifar_dataset_size)]
            background = img_as_ubyte(background)
            background = cv2.resize(background, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_AREA)
            self.bg_img = background

    def _get_pos_on_img(self, pos):
        return (self.boundary[0] + pos[0] * self.grid_size, self.boundary[1] + pos[1] * self.grid_size)

    def _draw_static_obj_and_bg(self):
        # canvas = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        self.canvas = self.bg_img.copy()
        for i, y in enumerate(self.opponents_y_pos):
            pos = self._get_pos_on_img((self.n_grid[0]-1, y))
            mask = self.opponents_digit[i].mask
            image = self.opponents_digit[i].image
            self.canvas[pos[0]: pos[0] + self.grid_size, pos[1]: pos[1] + self.grid_size][mask] = image[mask]

    def _draw_screen(self):
        screen = self.canvas.copy()
        pos = self._get_pos_on_img(self.agent_pos)
        mask = self.agent_digit.mask
        image = self.agent_digit.image
        screen[pos[0]: pos[0] + self.grid_size, pos[1]: pos[1] + self.grid_size][mask] = image[mask]
        return screen

    def _sample_a_digit(self, color=None):
        idx = self.np_rng.randint(self.mnist_dataset_size)
        image = self.mnist_dataset.data[idx].numpy()  # uint8
        label = self.mnist_dataset.targets[idx].item()  # int64

        image = cv2.resize(image, (self.grid_size, self.grid_size), interpolation=cv2.INTER_AREA)
        image[image < 10] = 0.0
        mask = image > 0
        if color is None:
            color = np.hstack([ self.np_rng.uniform(low=0.0, high=1.0, size=1),
                                self.np_rng.uniform(low=0.5, high=1.0, size=2)])
        image = np.tile(image[..., None], reps=[1, 1, 3])
        image = (image * color).round().astype(np.uint8)
        
        return Digit(image=image, label=label, mask=mask )

    def _create_new_agent(self):
        self.agent_pos = [0, self.np_rng.randint(0, self.n_grid[1]) ]
        self.agent_digit = self._sample_a_digit(self.agent_color)
        self.target_digit_labels = []
        digit_diff_list = [abs(oppo_digit.label - self.agent_digit.label) for oppo_digit in self.opponents_digit]
        min_diff = np.min(digit_diff_list)
        for i, diff in enumerate(digit_diff_list):
            if diff == min_diff:
                self.target_digit_labels.append(self.opponents_digit[i].label)
        
    def _create_new_opponents(self):
        self.opponents_digit = [self._sample_a_digit() for _ in range(self.n_opponents)]
        self.opponents_y_pos = []
        while len(self.opponents_y_pos) < self.n_opponents:
            y = self.np_rng.randint(0, self.n_grid[1])
            if y not in self.opponents_y_pos:
                self.opponents_y_pos.append(y)

    def _is_terminal(self):
        return (len(self.opponents_y_pos) == 0)

    def _get_info(self):
        # return None
        return {
            'agent_pos': self.agent_pos,
            'agent_label': self.agent_digit.label,
            'opponents_y_pos': self.opponents_y_pos.copy(),
            'opponents_label': [t.label for t in self.opponents_digit],
            'target_digit_labels': self.target_digit_labels,
        }

    def step(self, action):
        if self.num_steps_so_far >= self.max_ep_steps or self._is_terminal():
            return self._draw_screen(), 0.0, True, self._get_info()
        self.num_steps_so_far += 1

        self.agent_pos[0] += 1
        if self.agent_pos[0] >= self.n_grid[0]:
            self._create_new_agent()
            return self._draw_screen(), self.miss_reward, False, self._get_info()
        self.agent_pos[1] += self.action_to_y_pos_change[action]
        self.agent_pos[1] = max(0,min(self.n_grid[1]-1, self.agent_pos[1]))
        
        reward = self._manage_collision()
        done = self._is_terminal()

        return self._draw_screen(), reward, done, self._get_info()

    def _manage_collision(self):
        if self.agent_pos[0] != self.n_grid[0] - 1:
            return 0.0
        for i, y in enumerate(self.opponents_y_pos):
            if self.agent_pos[1] == y: # hit
                if self.opponents_digit[i].label in self.target_digit_labels:
                    reward = self.right_hit_reward
                    self.opponents_y_pos.pop(i)
                    self.opponents_digit.pop(i)
                    self._draw_static_obj_and_bg()
                    if self._is_terminal(): # don't need to create new agent
                        return reward
                else:
                    reward = self.wrong_hit_reward
                self._create_new_agent()
                return reward
        return 0.0

    def render(self, mode='rgb_array'):
        if mode != 'rgb_array':
            raise NotImplementedError()
        return self._draw_screen()

    def seed(self, x):
        self.np_rng = np.random.RandomState(x)