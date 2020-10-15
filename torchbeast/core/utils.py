#!/usr/bin/env python3

import torch
from typing import Union, List

def check_nan_and_inf(tensor_list: List[torch.Tensor]):
    for tensor in tensor_list:
        if torch.isnan(tensor).any():
            raise ValueError("found nan in tensor")