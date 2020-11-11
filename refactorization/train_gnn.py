#!/usr/bin/env python
import sys
import os
import os.path as osp

_ROOT_DIR = os.path.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, _ROOT_DIR)

import time
import socket
import warnings
from common.utils.logger import setup_logger

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

# torch.backends.cudnn.benchmark = True

from common.utils.checkpoint import CheckpointerV2
from common.utils.metric_logger import MetricLogger
from common.utils.torch_utils import set_random_seed

from refactorization.config.defaults_gnn import cfg
from common.utils.cfg_utils import purge_cfg
from refactorization.models_gnn.build import build_model
from refactorization.datasets.build import build_gnn_dataloader
from refactorization.train_utils import build_optimizer, build_lr_scheduler


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch 3D Deep Learning Training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--dev', action='store_true', help='develop mode')
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

    # run name
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs', 'outputs'))
        if args.dev:
            output_dir = osp.join(output_dir, run_name)
            warnings.warn('Dev mode enabled.')
        if osp.isdir(output_dir):
            warnings.warn('Output directory exists.')
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger('train', output_dir, filename='log.train.{:s}.txt'.format(run_name))
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    from common.utils.collect_env import collect_env_info
    logger.info('Collecting env info (might take some time)\n' + collect_env_info())

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    # ---------------------------------------------------------------------------- #
    # Build models, optimizer, scheduler, checkpointer, etc.
    # ---------------------------------------------------------------------------- #
    # build model
    set_random_seed(cfg.RNG_SEED)
    model = build_model(cfg)
    logger.info('Build model:\n{}'.format(str(model)))

    # Currently only support single-gpu mode
    model = model.cuda()

    # build optimizer
    optimizer = build_optimizer(cfg, model)

    # build lr scheduler
    lr_scheduler = build_lr_scheduler(cfg, optimizer)

    # build checkpointer
    # Note that checkpointer will load state_dict of model, optimizer and scheduler.
    checkpointer = CheckpointerV2(model,
                                  optimizer=optimizer,
                                  scheduler=lr_scheduler,
                                  save_dir=output_dir,
                                  logger=logger,
                                  max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data = checkpointer.load(cfg.RESUME_PATH,
                                        resume=cfg.AUTO_RESUME,
                                        resume_states=cfg.RESUME_STATES,
                                        strict=cfg.RESUME_STRICT)
    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD
    start_iter = checkpoint_data.get('iteration', 0)

    # build data loader
    # Reset the random seed again in case the initialization of models changes the random state.
    set_random_seed(cfg.RNG_SEED)
    train_dataloader = build_gnn_dataloader(cfg, True, start_iter)
    logger.info(train_dataloader.dataset)

    # build metrics
    train_meters = MetricLogger(delimiter='  ')

    def setup_train():
        model.train()
        train_meters.reset()

    # Build tensorboard logger
    summary_writer = None
    if output_dir:
        tb_dir = output_dir
        summary_writer = SummaryWriter(tb_dir, max_queue=64, flush_secs=30)

    # ---------------------------------------------------------------------------- #
    # Setup validation
    # ---------------------------------------------------------------------------- #
    val_period = cfg.VAL.PERIOD
    do_validation = val_period > 0
    if do_validation:
        val_dataloader = build_gnn_dataloader(cfg, training=False)
        logger.info(val_dataloader.dataset)
        val_meters = MetricLogger(delimiter='  ')

        best_metric_name = 'best_{}'.format(cfg.VAL.METRIC)
        best_metric = checkpoint_data.get(best_metric_name, None)

        def setup_validate():
            model.eval()
            val_meters.reset()

    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- #
    setup_train()
    max_iter = cfg.TRAIN.MAX_ITER
    logger.info('Start training from iteration {}'.format(start_iter))
    tic = time.time()

    for iteration, data_batch in enumerate(train_dataloader, start_iter):
        cur_iter = iteration + 1
        data_time = time.time() - tic

        # copy data from cpu to gpu
        data_batch = data_batch.to('cuda')

        # forward
        pd_dict = model(data_batch)

        # update losses
        loss_dict = model.compute_losses(pd_dict, data_batch, )
        total_loss = sum(loss_dict.values())

        # It is slightly faster to update metrics and meters before backward
        with torch.no_grad():
            train_meters.update(total_loss=total_loss, **loss_dict)
            model.update_metrics(pd_dict, data_batch, train_meters.metrics)

        # backward
        optimizer.zero_grad()
        total_loss.backward()
        if cfg.OPTIMIZER.MAX_GRAD_NORM > 0:
            # CAUTION: built-in clip_grad_norm_ clips the total norm.
            total_norm = clip_grad_norm_(model.parameters(), max_norm=cfg.OPTIMIZER.MAX_GRAD_NORM)
        else:
            total_norm = None
        optimizer.step()

        batch_time = time.time() - tic
        train_meters.update(time=batch_time, data=data_time)

        # log
        log_period = cfg.TRAIN.LOG_PERIOD
        if log_period > 0 and (cur_iter % log_period == 0 or cur_iter == 1):
            logger.info(
                train_meters.delimiter.join(
                    [
                        'iter: {iter:4d}',
                        '{meters}',
                        'lr: {lr:.2e}',
                        'max mem: {memory:.0f}',
                    ]
                ).format(
                    iter=cur_iter,
                    meters=str(train_meters),
                    lr=optimizer.param_groups[0]['lr'],
                    memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                )
            )

        # summary
        summary_period = cfg.TRAIN.SUMMARY_PERIOD
        if summary_writer is not None and (summary_period > 0 and cur_iter % summary_period == 0):
            keywords = ('loss', 'acc',)
            for name, metric in train_meters.metrics.items():
                if all(k not in name for k in keywords):
                    continue
                summary_writer.add_scalar('train/' + name, metric.result, global_step=cur_iter)

            # summarize gradient norm
            if total_norm is not None:
                summary_writer.add_scalar('grad_norm', total_norm, global_step=cur_iter)

        # ---------------------------------------------------------------------------- #
        # validate for one epoch
        # ---------------------------------------------------------------------------- #
        if do_validation and (cur_iter % val_period == 0 or cur_iter == max_iter):
            setup_validate()
            logger.info('Validation begins at iteration {}.'.format(cur_iter))

            start_time_val = time.time()
            tic = time.time()
            for iteration_val, data_batch in enumerate(val_dataloader):
                data_time = time.time() - tic

                # copy data from cpu to gpu
                data_batch = data_batch.to('cuda')

                # forward
                with torch.no_grad():
                    pd_dict = model(data_batch)

                # update losses and metrics
                loss_dict = model.compute_losses(pd_dict, data_batch)
                total_loss = sum(loss_dict.values())

                # update metrics and meters
                val_meters.update(loss=total_loss, **loss_dict)
                model.update_metrics(pd_dict, data_batch, val_meters.metrics)

                batch_time = time.time() - tic
                val_meters.update(time=batch_time, data=data_time)
                tic = time.time()

                if cfg.VAL.LOG_PERIOD > 0 and iteration_val % cfg.VAL.LOG_PERIOD == 0:
                    logger.info(
                        val_meters.delimiter.join(
                            [
                                'iter: {iter:4d}',
                                '{meters}',
                                'max mem: {memory:.0f}',
                            ]
                        ).format(
                            iter=iteration,
                            meters=str(val_meters),
                            memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                        )
                    )

            # END: validation loop
            epoch_time_val = time.time() - start_time_val
            logger.info('Iteration[{}]-Val {}  total_time: {:.2f}s'.format(
                cur_iter, val_meters.summary_str, epoch_time_val))

            # summary
            if summary_writer is not None:
                keywords = ('loss', 'acc', 'ap', 'recall')
                for name, metric in val_meters.metrics.items():
                    if all(k not in name for k in keywords):
                        continue
                    summary_writer.add_scalar('val/' + name, metric.result, global_step=cur_iter)

            # best validation
            if cfg.VAL.METRIC in val_meters.metrics:
                cur_metric = val_meters.metrics[cfg.VAL.METRIC].result
                if best_metric is None \
                        or (cfg.VAL.METRIC_ASCEND and cur_metric > best_metric) \
                        or (not cfg.VAL.METRIC_ASCEND and cur_metric < best_metric):
                    best_metric = cur_metric
                    checkpoint_data['iteration'] = cur_iter
                    checkpoint_data[best_metric_name] = best_metric
                    checkpointer.save('model_best', tag=False, **checkpoint_data)

            # restore training
            setup_train()

        # ---------------------------------------------------------------------------- #
        # After validation
        # ---------------------------------------------------------------------------- #
        # checkpoint
        if (ckpt_period > 0 and cur_iter % ckpt_period == 0) or cur_iter == max_iter:
            checkpoint_data['iteration'] = cur_iter
            if do_validation and best_metric is not None:
                checkpoint_data[best_metric_name] = best_metric
            checkpointer.save('model_{:06d}'.format(cur_iter), **checkpoint_data)

        # ---------------------------------------------------------------------------- #
        # Finalize one step
        # ---------------------------------------------------------------------------- #
        # since pytorch v1.1.0, lr_scheduler is called after optimization.
        if lr_scheduler is not None:
            lr_scheduler.step()
        tic = time.time()

    # END: training loop
    if do_validation and cfg.VAL.METRIC:
        logger.info('Best val-{} = {}'.format(cfg.VAL.METRIC, best_metric))


if __name__ == '__main__':
    main()
