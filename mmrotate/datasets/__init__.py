# Copyright (c) OpenMMLab. All rights reserved.
from .builder import build_dataset  # noqa: F401, F403
from .dior_r import DIORDataset
from .dota import DOTADataset  # noqa: F401, F403
from .dota2 import DOTA2Dataset
from .pipelines import *  # noqa: F401, F403

__all__ = ['DOTADataset', 'DOTA2Dataset', 'DIORDataset', 'build_dataset']
