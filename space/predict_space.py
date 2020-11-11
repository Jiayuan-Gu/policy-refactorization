#!/usr/bin/env python
from __future__ import division
import sys
import os
import os.path as osp

_ROOT_DIR = os.path.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, _ROOT_DIR)

import time
import socket
import warnings
from common.utils.logger import setup_logger

import pickle
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from common.utils.checkpoint import CheckpointerV2

from space.config.defaults import cfg
from common.utils.cfg_utils import purge_cfg
from space.models.build import build_model


def plot_results(image, boxes, labels=None, save_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(9, 9))
    ax.set_axis_off()
    ax.imshow(image)

    # detection
    for i, bbox in enumerate(boxes):
        # (x_min, y_min, x_max, y_max)
        rect = plt.Rectangle((bbox[0] - 0.5, bbox[1] - 0.5), (bbox[2] - bbox[0]), (bbox[3] - bbox[1]),
                             fill=False, edgecolor='yellow', linewidth=5.0)
        ax.add_patch(rect)
        if labels is not None:
            ax.text(bbox[0] - 0.5, bbox[1] - 0.5, str(labels[i]), color='yellow', fontsize=25.0)

    if save_path:
        # fig.savefig(save_path, bbox_inches='tight', pad_inches=1)
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    else:
        # plt.tight_layout(0, 0, 0)
        plt.show()
    plt.close(fig)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument('--ckpt-path', type=str, help='path to checkpoint file')
    parser.add_argument('--data-path', type=str, help='path to data')
    parser.add_argument('-o', '--output-filename', default=None, type=str, help='path to output')

    parser.add_argument('-b', '--batch-size', default=8, type=int, help='path to data')
    parser.add_argument('-l', '--log-period', default=100, type=int, help='period to log')
    parser.add_argument('--det-thresh', default=0.1, type=float)
    parser.add_argument('--vis-first-n', type=int, help='visualize')
    parser.add_argument('--vis-thresh', default=0.1, type=float)
    parser.add_argument('--output-dir', default='./data', type=str)

    args = parser.parse_args()
    return args


def main():
    # ---------------------------------------------------------------------------- #
    # Setup the experiment
    # ---------------------------------------------------------------------------- #
    args = parse_args()

    # load the configuration
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs', 'outputs'))
        if not osp.isdir(output_dir):
            warnings.warn('Make a new directory: {}'.format(output_dir))
            os.makedirs(output_dir)

    # run name
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)

    logger = setup_logger('predict', '', filename=f'log.predict.{run_name}.txt')
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    from common.utils.collect_env import collect_env_info
    logger.info('Collecting env info (might take some time)\n' + collect_env_info())

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    # ---------------------------------------------------------------------------- #
    # Setup the model and the dataset
    # ---------------------------------------------------------------------------- #
    batch_size = args.batch_size
    det_thresh = args.det_thresh
    vis_first_n = args.vis_first_n
    vis_thresh = args.vis_thresh

    # build model
    model = build_model(cfg)
    logger.info('Build model:\n{}'.format(str(model)))
    model = model.cuda()
    model.eval()

    # build checkpointer
    checkpointer = CheckpointerV2(model, save_dir=output_dir, logger=logger)

    if args.ckpt_path:
        # load weight if specified
        weight_path = args.ckpt_path.replace('@', output_dir)
        checkpointer.load(weight_path, resume=False)
    else:
        # load last checkpoint
        checkpointer.load(None, resume=True)

    # build dataset
    dataset_kwargs = dict(cfg.DATASET.VAL)
    dataset_kwargs.pop('start', None)
    dataset_kwargs.pop('end', None) # generate proposals for the whole dataset
    if cfg.DATASET.NAME == 'FallingDigit':
        from space.datasets.falling_digit import FallingDigit
        if args.data_path is not None:
            dataset_kwargs['path'] = args.data_path
        dataset = FallingDigit(to_tensor=True, **dataset_kwargs)
    else:
        raise ValueError('Unsupported dataset: {}.'.format(cfg.DATASET.NAME))
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=1,
                            )

    predictions = []
    if vis_first_n is not None:
        vis_dir = osp.join(output_dir, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
    else:
        vis_dir = None

    # ---------------------------------------------------------------------------- #
    # Inference
    # ---------------------------------------------------------------------------- #
    for batch_idx, data_batch in enumerate(dataloader):
        # copy data from cpu to gpu
        data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

        # forward
        with torch.no_grad():
            preds = model(data_batch, fast=True)

        boxes = preds['boxes'].cpu().numpy()  # (b, A * h1 * w1, 4)
        z_pres_p = preds['z_pres_p'].cpu().numpy()  # (b, A * h1 * w1)

        for sample_idx, boxes_per_image in enumerate(boxes):
            boxes_per_image = boxes_per_image.reshape(-1, 4)
            scores_per_image = z_pres_p[sample_idx]
            predictions_per_image = {}

            if det_thresh is not None:
                valid_mask = scores_per_image >= det_thresh
            else:
                valid_mask = np.ones_like(scores_per_image, dtype=bool)

            predictions_per_image['boxes'] = boxes_per_image[valid_mask]
            predictions_per_image['scores'] = scores_per_image[valid_mask]
            predictions.append(predictions_per_image)

            data_index = batch_idx * batch_size + sample_idx
            if vis_first_n is not None and (vis_first_n == -1 or data_index < vis_first_n):
                data = dataset.data[data_index]
                if 'image' in data:
                    image = data['image']
                else:
                    image = data['original_image']

                vis_path = osp.join(vis_dir, '{:06d}.png'.format(data_index))
                # vis_path = None
                vis_mask = scores_per_image >= vis_thresh

                plot_results(image, boxes_per_image[vis_mask],
                             labels=['{:.2f}'.format(x) for x in scores_per_image[vis_mask]],
                             save_path=vis_path)

        if args.log_period > 0 and batch_idx % args.log_period == 0:
            print(batch_idx, '/', len(dataloader))

    # save
    output_filename = args.output_filename
    if output_filename is None:
        output_filename = 'proposals_' + osp.basename(dataset.path)
    with open(osp.join(args.output_dir, output_filename), 'wb') as f:
        pickle.dump(predictions, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
