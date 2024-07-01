import copy
from typing import Dict, List, Optional, Tuple, Union

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from numpy import random
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import autocast_box_type
import torch


@TRANSFORMS.register_module()
class LoadCaption(BaseTransform):
    def __init__(self):
        super(LoadCaption, self).__init__()

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        # load caption
        if isinstance(results['captions'], str):
            captions = torch.load(results['captions'])
        else:
            captions = results['captions']
        results['captions'] = [captions]
        return results
