#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import numpy as np
import torch.multiprocessing as mp
from collections import deque
from skimage.io import imsave
# from ..utils import *
from ..utils.logger import Logger, GlobalLogger
from ..utils.normalizer import build_normalizer
from ..utils.misc import mkdir, close_obj, has_flag
from ..utils.torch_utils import random_seed
import logging

from ..component.envs import Task
from common.utils.checkpoint import CheckpointerV2_RL
from ..network.network_builder import build_network

import pickle

def update_dict_with_key_map(d1, d2, key_map):
    for k1, k2 in key_map.items():
        if k1 not in d1:
            raise Exception('k1 not in d1')
        if k2 not in d2:
            raise Exception('k2 not in d2')
        d1[k1] = d2[k2]

class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.logger = GlobalLogger(logging.getLogger('RL'), config.final_output_dir, 0)
        self.task_ind = 0
        self.state_normalizer = build_normalizer(config.RL.state_normalizer)
        self.reward_normalizer = build_normalizer(config.RL.reward_normalizer)
        self.checkpointer = None
        self.first_eval = True

    def close(self):
        close_obj(self.task)
        close_obj(self.evaluator)

    def lazy_init_checkpointer(self):
        if self.checkpointer is None:
            self.checkpointer = CheckpointerV2_RL(self.network,
                                                state_normalizer=self.state_normalizer,
                                                optimizer=self.optimizer,
                                                save_dir=self.config.final_output_dir,
                                                logger=self.logger,
                                                max_to_keep=self.config.train.n_checkpoints_to_keep)

    def save(self, tag=None):
        self.lazy_init_checkpointer()
        filename = '{:d}'.format(self.total_steps)
        if tag: filename += ('_' + tag) 
        self.checkpointer.save(filename)

    def try_to_load_network(self):
        config = self.config
        if config.load_ckpt:
            self.load(config.load_ckpt)

    def load(self, ckpt_path):
        self.lazy_init_checkpointer()
        self.checkpointer.load(ckpt_path, resume=False, resume_states=False)


    def eval_episodes(self):
        if self.config.eval.is_async and not self.first_eval:
            self.evaluator.query_eval_done() # let the training wait for evaluation
        self.first_eval = False
        self.evaluator.eval_episodes(self.total_steps)

    def record_online_return(self, info, offset=0):
        # pass
        if isinstance(info, dict):
            if 'episodic_return' in info: # wrapped by OriginalReturnWrapper
                ret = info['episodic_return']
            elif 'episode' in info: # procgen env
                ret = info['episode']['r']
            else:
                return
            if ret is not None:
                self.logger.add_scalar('episodic_return_train', ret, self.total_steps + offset)
                if not has_flag(self.config.train, 'hide_episodic_return'):
                    self.logger.info('steps %d, episodic_return_train %s' % (self.total_steps + offset, ret))
        elif isinstance(info, tuple) or isinstance(info, list):
            for i, info_ in enumerate(info):
                self.record_online_return(info_, i)
        else:
            raise NotImplementedError

    def switch_task(self):
        config = self.config
        # if not config.tasks:
        if not hasattr(config, 'tasks'):
            return
        segs = np.linspace(0, config.max_steps, len(config.tasks) + 1)
        if self.total_steps > segs[self.task_ind + 1]:
            self.task_ind += 1
            self.task = config.tasks[self.task_ind]
            self.states = self.task.reset()
            self.states = config.state_normalizer(self.states)

    def record_episode(self, dir, env):
        mkdir(dir)
        steps = 0
        state = env.reset()
        while True:
            self.record_obs(env, dir, steps)
            action = self.record_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            steps += 1
            if ret is not None:
                break

    def record_step(self, state):
        raise NotImplementedError

    # For DMControl
    def record_obs(self, env, dir, steps):
        env = env.env.envs[0]
        obs = env.render(mode='rgb_array')
        imsave('%s/%04d.png' % (dir, steps), obs)

# from ..component.envs import LazyFrames

class BaseActor(mp.Process):
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    NETWORK = 4
    CACHE = 5

    def __init__(self, config, lock):
        mp.Process.__init__(self)
        self.config = config
        self.lock = lock
        self.state_normalizer = build_normalizer(config.RL.state_normalizer)
        self.__pipe, self.__worker_pipe = mp.Pipe()

        self._state = None
        self._task = None
        self._network = None
        self._total_steps = 0
        self.__cache_len = 2

        if not config.DQN.async_actor:
            self.start = lambda: None
            self.step = self._sample
            self._set_up()
            # self._task = self.task_fn()
            self._task = self.build_task()

    def build_task(self):
        config = self.config
        return Task(config.task.full_name, **dict(config.other))

    def _sample(self):
        transitions = []
        for _ in range(self.config.DQN.sgd_update_frequency):
            transitions.append(self._transition())
        return transitions

    def run(self):
        self._set_up()
        config = self.config
        # self._task = self.task_fn()
        self._task = self.build_task()
        if hasattr(self.config.other, 'save_all_experience'):
            import h5py
            self.h5_data = h5py.File(self.config.other.save_all_experience, mode='w')

        cache = deque([], maxlen=2)
        while True:
            op, data = self.__worker_pipe.recv()
            if op == self.STEP:
                if not len(cache):
                    cache.append(self._sample())
                    cache.append(self._sample())
                self.__worker_pipe.send(cache.popleft())
                cache.append(self._sample())
            elif op == self.EXIT:
                self.__worker_pipe.close()
                if hasattr(self.config.other, 'save_all_experience'):
                    self.h5_data.close()
                    print('@@@@@@@@@@@@@@@@ close h5')
                return
            elif op == self.NETWORK:
                self._network = data
            else:
                raise NotImplementedError

    def _transition(self):
        raise NotImplementedError

    def _set_up(self):
        pass

    def step(self):
        self.__pipe.send([self.STEP, None])
        return self.__pipe.recv()

    def close(self):
        if self.config.DQN.async_actor:
            self.__pipe.send([self.EXIT, None])
            self.__pipe.close()

    def set_network(self, net):
        if not self.config.DQN.async_actor:
            self._network = net
        else:
            self.__pipe.send([self.NETWORK, net])


from ..utils.logger import EvalResultsWriter

class BaseEvaluator(mp.Process):
    EVAL = 0
    EXIT = 1
    NETWORK = 2
    # LOG = 3

    def __init__(self, config, lock, logger):
        mp.Process.__init__(self)
        self.config = config
        self.lock = lock
        self.logger = logger
        self.state_normalizer = build_normalizer(config.RL.state_normalizer)
        
        if config.eval.is_async:
            self.__pipe, self.__worker_pipe = mp.Pipe()
            self.task = None
            self.network_outside = None # this is just a handle
            self.network = None
        else:
            self.start = lambda: None
            self.close = lambda: None
            self.eval_episodes = self._eval_episodes
            self._set_up()
            # self.task = self.task_fn()
            self.task = self.build_task()
            self.network = build_network(config)
            # self.results_writer = self.results_writer_fn()
            self.results_writer = self.build_writer()

    def build_task(self):
        config = self.config
        return Task(config.task.full_name, 
                                    num_envs=config.eval.n_episodes if config.eval.parallel else 1, 
                                    single_process=not config.eval.env_subprocess,
                                    **dict(config.other))

    def build_writer(self):
        config = self.config
        return EvalResultsWriter('{:s}/eval'.format(config.final_output_dir) if config.final_output_dir else None, 
                                        header={'env_id' : config.task.full_name})

    def run(self):
        self._set_up()
        random_seed()
        config = self.config
        # self.task = self.task_fn()
        self.task = self.build_task()
        self.network = build_network(config)
        # self.results_writer = self.results_writer_fn()
        self.results_writer = self.build_writer()

        while True:
            op, data = self.__worker_pipe.recv()
            if op == self.EVAL:
                eval_done = self._eval_episodes(data)
                self.__worker_pipe.send(eval_done)
                # steps, mean, std = self._eval_episodes(data)
                # self.__worker_pipe.send((steps, mean, std))
            elif op == self.EXIT:
                self.__worker_pipe.close()
                return
            elif op == self.NETWORK:
                self.network_outside = data
            else:
                raise NotImplementedError

    def _set_up(self):
        pass

    def close(self):
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()

    def set_network(self, net):
        if not self.config.eval.is_async:
            self.network_outside = net
        else:
            self.__pipe.send([self.NETWORK, net])

    def query_eval_done(self):
        eval_done = self.__pipe.recv()
        return eval_done

    def eval_episodes(self, current_steps):
        self.__pipe.send([self.EVAL, current_steps])

    def eval_single_episode(self):
        env = self.task
        state = env.reset()
        while True:
            action = self.eval_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            if ret is not None:
                break
        return ret

    def _eval_episodes(self, steps):
        with self.lock: # copy the network weight
            self.network.load_state_dict(self.network_outside.state_dict())
        self.network.eval()
        if self.config.eval.parallel:
            episodic_returns = self.eval_episode_parallel()
        else:
            episodic_returns = self.eval_episode_sequential()
        # print('@@@@@@@@@@@@@@@@@@@@ eval done')
        self.logger.info('steps %d, *** episodic_return_test %.3f (std = %.2f)' % (
            steps, np.mean(episodic_returns), np.std(episodic_returns)
        ))
        self.logger.add_scalar('episodic_return_test', np.mean(episodic_returns), steps)
        self.results_writer.write_row(steps, episodic_returns)
        return True
        # return steps, np.mean(episodic_returns), np.std(episodic_returns)

    def eval_episode_parallel(self):
        episodic_returns = [ None for _ in range(self.config.eval.n_episodes) ]
        done_cnt = 0
        env = self.task
        state = env.reset()
        step_cnt = 0
        while True:
            step_cnt += 1
            action = self.eval_step(state)
            state, reward, done, info = env.step(action)
            for i_env, _info in enumerate(info):
                ret = _info['episodic_return']
                if episodic_returns[i_env] is None and ret is not None:
                    episodic_returns[i_env] = ret
                    done_cnt += 1
                    if done_cnt >= self.config.eval.n_episodes:
                        # print('@@@@@@@@ eval step cnt:', step_cnt)
                        return episodic_returns

    def eval_episode_sequential(self):
        episodic_returns = []
        for ep in range(self.config.eval.n_episodes):
            total_rewards = self.eval_single_episode()
            episodic_returns.append(np.sum(total_rewards))
        return episodic_returns

    def eval_step(self, state):
        raise NotImplementedError
