import torch
from torch.utils.data import Sampler


class PartitionRandomSampler(Sampler):
    r"""Samples elements randomly from within sequential partitions/ slices, without replacement.

    Arguments:
        partition sizes (sequence): a sequence of partition sizes
    """

    def __init__(self, partition_start_end_indices):
        self.partition_start_end_indices = partition_start_end_indices
        partition_end_indices = [end_idx for (_, end_idx) in self.partition_start_end_indices]
        self.num_indices = max(partition_end_indices) + 1

    def __iter__(self):
        randomized_indices = []
        for (start_idx, end_idx) in self.partition_start_end_indices:
            rand_indices_in_partition = start_idx + torch.randperm(end_idx - start_idx + 1)
            randomized_indices.extend(rand_indices_in_partition)
        return iter(randomized_indices)

    def __len__(self):
        return self.num_indices