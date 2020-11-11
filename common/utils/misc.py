"""Miscellaneous functions.

Notes:
    What kind of functions should be included here?
    1. Simple (a few lines) but widely used.
    2. For debugging
"""

import os
import numpy as np
import torch
import gzip
import pickle


def print_dict(d: dict):
    """Print the given dictionary for debugging."""
    for k, v in d.items():
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, v.shape)
        else:
            print(k, v)


def dynamic_load_modules(target_dir, context,
                         excludes=('__init__.py',),
                         verbose=True):
    """Load all the python files(.py) in the current directory.

    Notes:
        It is suggested to import modules explicitly.
        However, sometimes, we may temporarily add some modules to experiments,
        but do not want to include them in git or hope the program can still run
        when we remove the experimental modules.
        "from xxx import *" is not encouraged, unless __all__ is controlled carefully.

    """
    all_filenames = os.listdir(target_dir)
    py_filenames = [x for x in all_filenames if x.endswith('.py') and x not in excludes]
    if verbose:
        print(py_filenames)
    module_names = [os.path.splitext(x)[0] for x in py_filenames]
    for name in module_names:
        exec('from .{} import *'.format(name), context)


def dump_pickle(obj, path):
    if path.endswith('.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    elif path.endswith('.pgz'):
        with gzip.open(path, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    else:
        raise RuntimeError('Unsupported extension {}.'.format(os.path.splitext(path)[-1]))


def load_pickle(path):
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)
    elif path.endswith('.pgz'):
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    else:
        raise RuntimeError('Unsupported extension {}.'.format(os.path.splitext(path)[-1]))
