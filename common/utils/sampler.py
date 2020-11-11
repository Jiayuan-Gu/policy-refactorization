import itertools
from torch.utils.data.sampler import Sampler


class IterationBasedBatchSampler(Sampler):
    """Wraps a BatchSampler.

    Resampling from it until a specified number of iterations have been sampled

    References:
        https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration < self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                yield batch
                iteration += 1
                if iteration >= self.num_iterations:
                    break

    def __len__(self):
        return self.num_iterations - self.start_iter


class RepeatSampler(Sampler):
    def __init__(self, data_source, repeats=1):
        self.data_source = data_source
        self.repeats = repeats

    def __iter__(self):
        return iter(itertools.chain(*[range(len(self.data_source))] * self.repeats))

    def __len__(self):
        return len(self.data_source) * self.repeats


def test_IterationBasedBatchSampler():
    from torch.utils.data.sampler import SequentialSampler, BatchSampler
    sampler = SequentialSampler([i for i in range(10)])
    batch_sampler = BatchSampler(sampler, batch_size=2, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, 5)

    # Check __len__
    assert len(batch_sampler) == 5
    for i, index in enumerate(batch_sampler):
        assert [i * 2, i * 2 + 1] == index

    # Check start iter
    batch_sampler.start_iter = 2
    assert len(batch_sampler) == 3


def test_RepeatSampler():
    data_source = [1, 2, 5, 3, 4]
    repeats = 5
    sampler = RepeatSampler(data_source, repeats=repeats)
    assert len(sampler) == repeats * len(data_source)
    sampled_indices = list(iter(sampler))
    assert sampled_indices == list(range(len(data_source))) * repeats
