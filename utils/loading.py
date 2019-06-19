import torch
from torch.utils.data import SubsetRandomSampler
from datasets.partition_random_sampler import PartitionRandomSampler


def data_loader_for_dataset(dataset, batch_size, shuffle):
    # pin_memory = device == 'cuda'
    pin_memory = False
    if shuffle:
        start_end_indices = dataset.get_sample_start_and_end_indices_per_file()
        sampler = PartitionRandomSampler(start_end_indices)

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,
            sampler=sampler, pin_memory=pin_memory,
            num_workers=0  # use main worker
        )
    else:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,
            pin_memory=pin_memory, num_workers=0  # use main worker
        )

    return loader
