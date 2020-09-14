from detectron2.config import CfgNode as CN


def add_center_config(cfg):
    """
    Add config for CenterMask.
    """
    _C = cfg

    _C.MODEL.SEMANTIC_RELATION = CN()
    _C.MODEL.SEMANTIC_RELATION.ETA = 0.4
    _C.MODEL.SEMANTIC_RELATION.GAMMA = 0.2
    _C.MODEL.SEMANTIC_RELATION.TAU = 0.5
