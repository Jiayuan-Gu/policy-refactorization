import warnings

from .ec_net import EdgeConvNet


def build_model(cfg):
    model_fn = globals()[cfg.MODEL.TYPE]
    if cfg.MODEL.TYPE in cfg.MODEL:
        model_cfg = dict(cfg.MODEL[cfg.MODEL.TYPE])
    else:
        warnings.warn('Use default arguments to initialize {}'.format(cfg.MODEL.TYPE))
        model_cfg = dict()
    model = model_fn(**model_cfg)
    return model
