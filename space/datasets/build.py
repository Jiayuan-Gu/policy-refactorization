from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader
from common.utils.torch_utils import worker_init_fn
from common.utils.sampler import IterationBasedBatchSampler


def build_dataset(cfg, training=True):
    dataset_kwargs = cfg.DATASET.get('TRAIN' if training else 'VAL')

    if cfg.DATASET.NAME == 'FallingDigit':
        from .falling_digit import FallingDigit
        dataset = FallingDigit(to_tensor=True, **dataset_kwargs)
    else:
        raise ValueError('Unsupported dataset: {}.'.format(cfg.DATASET.NAME))

    return dataset


def build_dataloader(cfg, training=True, start_iter=0):
    dataset = build_dataset(cfg, training=training)
    worker_seed = cfg.RNG_SEED if cfg.RNG_SEED >= 0 else None

    if training:
        sampler = RandomSampler(dataset, replacement=False)
        batch_sampler = BatchSampler(sampler, batch_size=cfg.TRAIN.BATCH_SIZE, drop_last=True)
        batch_sampler = IterationBasedBatchSampler(batch_sampler,
                                                   num_iterations=cfg.TRAIN.MAX_ITER,
                                                   start_iter=start_iter)
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=worker_seed),
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.VAL.BATCH_SIZE,
            shuffle=True,  # For visualization
            # shuffle=False,
            drop_last=False,
            num_workers=cfg.VAL.NUM_WORKERS,
            worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=worker_seed),
        )
    return dataloader
