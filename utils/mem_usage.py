import gc
import sys

import torch
from run_scripts import memory_logging


def print_torch_mem_usage(desc, print_mem=True):
    if not memory_logging:
        return None
    total_size_torch = 0.0
    total_size_other = 0.0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                obj_size = 4.0  # in bytes
                for s in obj.size():
                    obj_size *= s
                total_size_torch += obj_size / (1000*1000)  # MB
            elif sys.getsizeof(obj) > 0:
                total_size_other += sys.getsizeof(obj) / (1000*1000)  # MB

        except:
            pass
    if print_mem:
        print('-- mem usage {}: {:10.4f} MB (torch) + {:10.4f} MB (other) --'.format(desc, total_size_torch, total_size_other))
        print()
        sys.stdout.flush()
    return total_size_torch
