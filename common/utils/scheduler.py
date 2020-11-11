from __future__ import division
from bisect import bisect_right
import math

__all__ = ['MultiStepScheduler', 'LinearScheduler', 'LogLinearScheduler']


class MultiStepScheduler(object):
    def __init__(self, initial_value, values, milestones):
        self.values = (initial_value,) + tuple(values)
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )
        self.milestones = milestones
        assert len(self.milestones) + 1 == len(self.values)

    def __call__(self, epoch):
        return self.values[bisect_right(self.milestones, epoch)]

    def __str__(self):
        attr_list = ['values', 'milestones']
        attr_str = ', '.join(['{}={}'.format(k, getattr(self, k)) for k in attr_list])
        return '{}({})'.format(self.__class__.__name__, attr_str)


class LinearScheduler(object):
    def __init__(self, values, milestones, interval=1):
        assert len(values) == len(milestones) == 2
        assert milestones[0] < milestones[1]
        self.values = values
        self.milestones = milestones
        self.interval = interval

    def __call__(self, epoch):
        if epoch <= self.milestones[0]:
            return self.values[0]
        elif epoch >= self.milestones[1]:
            return self.values[1]
        else:
            epoch_residue = (epoch - self.milestones[0]) % self.interval
            epoch_floor = epoch - epoch_residue
            ratio = (epoch_floor - self.milestones[0]) / (self.milestones[1] - self.milestones[0])
            return (1.0 - ratio) * self.values[0] + ratio * self.values[1]

    def __str__(self):
        attr_list = ['values', 'milestones', 'interval']
        attr_str = ', '.join(['{}={}'.format(k, getattr(self, k)) for k in attr_list])
        return '{}({})'.format(self.__class__.__name__, attr_str)


class LogLinearScheduler(object):
    def __init__(self, values, milestones, interval=1):
        assert len(values) == len(milestones) == 2
        assert milestones[0] < milestones[1]
        self.values = values
        self.milestones = milestones
        self.interval = interval

    def __call__(self, epoch):
        if epoch <= self.milestones[0]:
            return self.values[0]
        elif epoch >= self.milestones[1]:
            return self.values[1]
        else:
            epoch_residue = (epoch - self.milestones[0]) % self.interval
            epoch_floor = epoch - epoch_residue
            ratio = (epoch_floor - self.milestones[0]) / (self.milestones[1] - self.milestones[0])
            if ratio < 1e-12:
                return self.values[0]
            else:
                log_ratio = ratio * math.log(self.values[1] / self.values[0])
                return self.values[0] * math.exp(log_ratio)

    def __str__(self):
        attr_list = ['values', 'milestones', 'interval']
        attr_str = ', '.join(['{}={}'.format(k, getattr(self, k)) for k in attr_list])
        return '{}({})'.format(self.__class__.__name__, attr_str)


def test_MultiStepScheduler():
    target = [1.0, 0.2, 0.3, 0.5, 0.5]
    scheduler = MultiStepScheduler(1.0, values=[0.2, 0.3, 0.5], milestones=[1, 2, 3])
    output = []
    for i in range(len(target)):
        output.append(scheduler(i))
    assert target == output


def test_LinearScheduler():
    target = [1.0, 1.0, 0.5, 0.0, 0.0]
    scheduler = LinearScheduler([1.0, 0.0], [1, 3])
    output = []
    for i in range(len(target)):
        output.append(scheduler(i))
    import numpy as np
    np.testing.assert_allclose(output, target)


def test_LogLinearScheduler():
    target = [1.0, 1.0, 0.5, 0.25, 0.25]
    scheduler = LogLinearScheduler([1.0, 0.25], [1, 3])
    output = []
    for i in range(len(target)):
        output.append(scheduler(i))
    import numpy as np
    np.testing.assert_allclose(output, target)
