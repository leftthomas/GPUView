from detectron2.config import CfgNode as CN


def add_center_config(cfg):
    """
    Add config for CenterMask.
    """
    _C = cfg

    _C.MODEL.CENTER_MASK = CN()
    _C.MODEL.CENTER_MASK.S = 32
    _C.MODEL.CENTER_MASK.LAMBDA_P = 1.0
    _C.MODEL.CENTER_MASK.LAMBDA_OFF = 1.0
    _C.MODEL.CENTER_MASK.LAMBDA_SIZE = 0.1
    _C.MODEL.CENTER_MASK.LAMBDA_MASK = 1.0
