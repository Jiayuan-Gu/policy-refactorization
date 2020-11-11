#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import time
from collections import defaultdict
import numpy as np
import torch
import torch.multiprocessing as mp

# from ..network import *
from ..network.network_builder import build_network
import torch.nn as nn
# from ..component import *
from ..component.replay import build_replay
# from ..utils import *
from ..utils.misc import close_obj, has_flag, print_dict
from ..utils.torch_utils import build_optimizer, range_tensor, tensor, to_np
from ..utils.schedule import build_schedule
# from .BaseAgent import *
from .BaseAgent import BaseAgent, BaseActor, BaseEvaluator

import sys
from ..component.envs import LazyFrames

import torch.nn.functional as F
# import torchvision.transforms as T

class DQNActor(BaseActor):
    def __init__(self, config, lock):
        BaseActor.__init__(self, config, lock)
        self.config = config
        self.random_action_prob = build_schedule(config.DQN.e_greedy)
        self.time_stat = defaultdict(float)
        if config.DQN.async_actor:
            self.start()

    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()
        self._state_np = np.expand_dims(self._state[0].__array__(), axis=0) # manually convert to np array
        with self.lock:
            # q_values = self._network(config.state_normalizer(self._state))
            q_values = self._network(self.state_normalizer(self._state_np))
        q_values = to_np(q_values).flatten()
        if self._total_steps < self.config.DQN.exploration_steps \
                or np.random.rand() < self.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
        next_state, reward, done, info = self._task.step([action])
        entry = [self._state[0], action, reward[0], next_state[0], int(done[0]), info]
        self._total_steps += 1
        self._state = next_state
        return entry


class DQNEvaluator(BaseEvaluator):
    def __init__(self, config, lock, logger):
        BaseEvaluator.__init__(self, config, lock, logger)
        self.config = config
        if config.eval.is_async:
            self.start()

    def eval_step(self, state):
        self.state_normalizer.set_read_only()
        # state_np = np.expand_dims(state[0].__array__(), axis=0) # manually convert to np array
        state_np = np.stack([x.__array__() for x in state], axis=0) # manually convert to np array
        q = self.network(self.state_normalizer(state_np))
        action = to_np(q.argmax(-1))
        self.state_normalizer.unset_read_only()
        return action

    def eval_q(self, state):
        self.state_normalizer.set_read_only()
        state_np = np.stack([x.__array__() for x in state], axis=0) # manually convert to np array
        state_np = self.state_normalizer(state_np)
        q = self.network(state_np)
        q = to_np(q).flatten()
        self.state_normalizer.unset_read_only()
        return q

    def eval_q_from_processed_image(self, state):
        q = self.network(np.expand_dims(state, axis=0))
        q = to_np(q).flatten()
        return q
    

class DQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.lock = mp.Lock()

        # self.replay = config.replay_fn()
        self.replay = build_replay(config.DQN.replay)
        self.actor = DQNActor(config, self.lock)
        self.evaluator = DQNEvaluator(config, self.lock, self.logger)
        # self.evaluator = DQNEvaluator(config, self.lock)

        self.network = build_network(config)
        self.network.share_memory()
        self.optimizer = build_optimizer(config, self.network.parameters())
        self.try_to_load_network() # MUST BEFORE TARGET NET!!!!!!!!!!!!!
        
        self.target_network = build_network(config)
        self.target_network.load_state_dict(self.network.state_dict())

        self.actor.set_network(self.network)
        self.evaluator.set_network(self.network)

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size)
        self.time_stat = defaultdict(float)

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)
        close_obj(self.evaluator)

    def step(self):
        t = []
        t.append(time.time())
        config = self.config
        transitions = self.actor.step()
        t.append(time.time())

        experiences = []
        for state, action, reward, next_state, done, info in transitions:
            self.record_online_return(info)
            self.total_steps += 1
            reward = self.reward_normalizer(reward)
            experiences.append([state, action, reward, next_state, done])
        self.replay.feed_batch(experiences)
        t.append(time.time())

        if self.total_steps > self.config.DQN.exploration_steps:
        # if self.total_steps > 10000:
            for _ in range(config.DQN.n_batch_per_update):
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals = experiences

                states = tensor(self.state_normalizer(states))
                next_states = tensor(self.state_normalizer(next_states))
                t.append(time.time())
                q_next = self.target_network(next_states)
                q_next = q_next.detach()
                if self.config.DQN.double_q:
                    best_actions = torch.argmax(self.network(next_states), dim=-1)
                    q_next = q_next[self.batch_indices, best_actions]
                else:
                    q_next = q_next.max(1)[0]
                terminals = tensor(terminals)
                rewards = tensor(rewards)
                q_next = self.config.RL.discount * q_next * (1 - terminals)
                q_next.add_(rewards)
                actions = tensor(actions).long()
                q = self.network(states)
                q = q[self.batch_indices, actions]
                loss = (q_next - q).pow(2).mul(0.5).mean()
                t.append(time.time())
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.RL.gradient_clip)
                with self.lock:
                    self.optimizer.step()
                t.append(time.time())

        if self.total_steps / self.config.DQN.sgd_update_frequency % \
                self.config.DQN.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
