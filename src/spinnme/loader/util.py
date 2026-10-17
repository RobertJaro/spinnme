

import numpy as np
import torch


class MultiprocessingWrapper:

    def __init__(self, func, unpack='dict', **kwargs):
        self.func = func
        self.kwargs = kwargs
        self.unpack = unpack

    def run(self, input):
        if isinstance(input, dict) and '_preparation_seed' in input:
            input = input.copy()
            seed = input.pop('_preparation_seed')
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
        if self.unpack == 'list':
            return self.func(*input, **self.kwargs)
        elif self.unpack == 'dict':
            return self.func(**input, **self.kwargs)
        else:
            return self.func(input, **self.kwargs)
