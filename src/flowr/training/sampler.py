import math
import random

import torch.distributed as dist
from torch.utils.data import Sampler

from flowr.training.dataset import Dataset


class DistributedBucketSampler(Sampler):
    """This sampler partitions the batches in terms of datasets first, and then across processes based on rank and world_size."""

    def __init__(self, dataset: Dataset, batch_size: int, shuffle=True, rank=None, world_size=None):
        """
        Args:
            dataset (Dataset): Dataset.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle batches.
            rank (int, optional): Rank of the current process.
            world_size (int, optional): Total number of processes.
        """
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if world_size is None:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Bucket indices based on data roots.
        self.buckets = dataset.buckets

        # Shuffle each bucket.
        if self.shuffle:
            for bucket in self.buckets:
                random.shuffle(self.buckets[bucket])

        # Partition each bucket into batches.
        self.batches = []
        for bucket in self.buckets:
            bucket_indices = self.buckets[bucket]
            for i in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[i : i + self.batch_size]
                if len(batch) > 0:
                    self.batches.append(batch)

        if self.shuffle:
            random.shuffle(self.batches)

        # Make sure the total number of batches is divisible by world_size.
        self.num_batches = len(self.batches)
        self.num_samples = int(math.ceil(self.num_batches * 1.0 / self.world_size))
        self.total_size = self.num_samples * self.world_size

    def __iter__(self):
        # If necessary, duplicate some batches to make total_size.
        batches = list(self.batches)
        if len(batches) < self.total_size:
            extra = self.total_size - len(batches)
            batches.extend(batches[:extra])

        # Subsample: each process gets batches starting from its rank
        indices = list(range(len(batches)))[self.rank : self.total_size : self.world_size]
        for i in indices:
            yield batches[i]

    def __len__(self):
        return self.num_samples
