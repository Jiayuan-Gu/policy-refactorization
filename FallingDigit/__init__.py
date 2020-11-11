from .falling_digit import FallingDigitEnv
from gym.envs.registration import register

for i in range(1, 10):
    register(
        id='FallingDigitBlack_{:d}-v1'.format(i),
        entry_point='FallingDigit:FallingDigitEnv',
        kwargs={'n_opponents':i, 'n_instance_per_digit': 1, 'num_levels':3000 },
    )
    register(
        id='FallingDigitCIFAR_{:d}-v1'.format(i),
        entry_point='FallingDigit:FallingDigitEnv',
        kwargs={'n_opponents':i, 'bg_type': 'cifar', 'num_levels':3000, 'n_instance_per_digit': 1, 'n_bg_images': 100 },
    )
    register(
        id='FallingDigitBlack_{:d}_test-v1'.format(i),
        entry_point='FallingDigit:FallingDigitEnv',
        kwargs={'n_opponents':i, 'n_instance_per_digit': 1, 'start_level':3000, 'num_levels': 10000 },
    )
    register(
        id='FallingDigitCIFAR_{:d}_test-v1'.format(i),
        entry_point='FallingDigit:FallingDigitEnv',
        kwargs={'n_opponents':i, 'bg_type': 'cifar', 'start_level':3000, 'num_levels': 10000, 'n_instance_per_digit': 1, 'n_bg_images': 100 },
    )