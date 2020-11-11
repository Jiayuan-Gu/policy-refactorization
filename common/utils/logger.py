# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified by Jiayuan Gu
import logging
import os
import sys


def setup_logger(name, save_dir, filename='log.txt'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        log_path = os.path.join(save_dir, filename)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # To avoid conflict with Tensorflow
    # https://stackoverflow.com/questions/33662648/tensorflow-causes-logging-messages-to-double
    logger.propagate = False

    return logger
