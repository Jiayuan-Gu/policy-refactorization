#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from tensorboardX import SummaryWriter
import os
import os.path as osp
import numpy as np
import torch
import logging

# logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s')

import sys

def setup_logger(name, save_dir, comment=''):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        filename = 'log'
        if comment:
            filename += '.' + comment
        log_file = os.path.join(save_dir, filename + '.txt')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

class Logger(object):
    def __init__(self, vanilla_logger, log_dir, log_level=0):
        self.log_level = log_level
        self.writer = None
        if vanilla_logger is not None:
            self.info = vanilla_logger.info
            self.debug = vanilla_logger.debug
            self.warning = vanilla_logger.warning
        self.all_steps = {}
        self.log_dir = log_dir

    def lazy_init_writer(self):
        if self.log_dir and self.writer is None:
            self.writer = SummaryWriter(self.log_dir, flush_secs=30)

    def to_numpy(self, v):
        if isinstance(v, torch.Tensor):
            v = v.cpu().detach().numpy()
        return v

    def get_step(self, tag):
        if tag not in self.all_steps:
            self.all_steps[tag] = 0
        step = self.all_steps[tag]
        self.all_steps[tag] += 1
        return step

    def add_scalar(self, tag, value, step=None, log_level=0):
        if not self.log_dir:
            return
        self.lazy_init_writer()
        if log_level > self.log_level:
            return
        value = self.to_numpy(value)
        if step is None:
            step = self.get_step(tag)
        if np.isscalar(value):
            value = np.asarray([value])
        self.writer.add_scalar(tag, value, step)

    def add_histogram(self, tag, values, step=None, log_level=0):
        if not self.log_dir:
            return
        self.lazy_init_writer()
        if log_level > self.log_level:
            return
        values = self.to_numpy(values)
        if step is None:
            step = self.get_step(tag)
        self.writer.add_histogram(tag, values, step)

    def flush(self):
        if not self.log_dir:
            return
        self.writer.flush()


from tensorboardX import GlobalSummaryWriter

class tmuGlobalSummaryWriter(GlobalSummaryWriter):
    def __init__(self, *args, **kwargs):
        super(tmuGlobalSummaryWriter, self).__init__(*args, **kwargs)

    def add_scalar(self, tag, scalar_value, global_step, walltime=None):
        """add scalar with given global step"""
        with self.lock:
            self.smw.add_scalar(tag, scalar_value, global_step, walltime)

class GlobalLogger(Logger):
    def __init__(self, vanilla_logger, log_dir, log_level=0):
        super(GlobalLogger, self).__init__(vanilla_logger, log_dir)
        self.add_histogram = None
        if log_dir:
            self.writer = tmuGlobalSummaryWriter(self.log_dir, flush_secs=30)

    # def lazy_init_writer(self):
    #     if self.log_dir and self.writer is None:
    #         self.writer = tmuGlobalSummaryWriter.getSummaryWriter(self.log_dir)

    def add_scalar(self, tag, value, step=None, log_level=0):
        if not self.log_dir:
            return
        # self.lazy_init_writer()
        if log_level > self.log_level:
            return
        value = self.to_numpy(value)
        if step is None:
            step = self.get_step(tag)
        if np.isscalar(value):
            value = np.asarray([value])
        self.writer.add_scalar(tag, value, step)

import json, csv

class EvalResultsWriter(object):
    def __init__(self, filename, header=''):
        if not filename:
            return
        EXT = 'csv'
        if not filename.endswith(EXT):
            if osp.isdir(filename):
                filename = osp.join(filename, EXT)
            else:
                filename = filename + "." + EXT
        self.f = open(filename, "wt")
        if isinstance(header, dict):
            header = '# {} \n'.format(json.dumps(header))
        self.f.write(header)
        self.logger = csv.writer(self.f)
        self.logger.writerow(['step', 'ep_rewards'])
        self.f.flush()

    def write_row(self, step, ep_rewards):
        if hasattr(self, 'logger'):
            self.logger.writerow([step] + ep_rewards)
            self.f.flush()