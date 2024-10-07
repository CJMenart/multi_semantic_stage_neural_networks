"""
A Distributed batch sampler that combines supervised and partially supervised data, sampling from each in a controlled manner.
"""
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
import math
import torch
from typing import Optional, Iterator
import torch.distributed as dist


class ExtraCityscapesSampler(Sampler):

    def __init__(self, dataset: Dataset, nsupervised: int, batch_size: int, num_replicas: Optional[int] = None,
            rank: Optional[int] = None, seed: int = 0) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.nsupervised = nsupervised
        self.batch_size = batch_size
        
        self.num_samples = math.ceil(
                (nsupervised  / (self.num_replicas*self.batch_size)) # type: ignore[arg-type]
            ) * 2
        self.seed = seed

    def __iter__(self):
        return iter(self.build_batches())

    def build_batches(self):
        total_sz = len(self.dataset)
        supervised_ind = list(range(self.nsupervised))
        unsupervised_ind = list(range(self.nsupervised, total_sz))
            
        supervised_ind = self.shuffle(supervised_ind)
        unsupervised_ind = self.shuffle(unsupervised_ind)
        
        effective_batch_size = self.batch_size * self.num_replicas
        padding_size = len(supervised_ind) % effective_batch_size
        supervised_ind += supervised_ind[:padding_size]
        unsupervised_ind = unsupervised_ind[:len(supervised_ind)]

        supervised_ind = supervised_ind[self.rank:len(supervised_ind):self.num_replicas]
        unsupervised_ind = unsupervised_ind[self.rank:len(unsupervised_ind):self.num_replicas]
        
        batches = []
        for i in range(int(len(supervised_ind)/self.batch_size)):
            batches.append(supervised_ind[i*self.batch_size:(i+1)*self.batch_size])
            batches.append(unsupervised_ind[i*self.batch_size:(i+1)*self.batch_size])
        batches = self.shuffle(batches)
        return batches

    def shuffle(self, t):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        t = torch.tensor(t)
        indices = torch.randperm(t.shape[0], generator=g).tolist()  # type: ignore[arg-type]
        return t[indices].tolist()

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
