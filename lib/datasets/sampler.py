from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler
from torch.distributed import get_rank, get_world_size


class DistributedClassBalancedSubsetSampler(Sampler):

    def __init__(self, dataset, indices):
        self.rank = get_rank()
        self.num_replicas = get_world_size()
        self.cat2inds = defaultdict(list)
        for ind in indices:
            self.cat2inds[(dataset.get_cat_ids(ind))].append(ind)
        self.max_num_inds = max(len(_) for _ in self.cat2inds.values())
        self.max_num_inds = int(np.ceil(
            self.max_num_inds / self.num_replicas)) * self.num_replicas
        self.epoch = 0

    def __iter__(self):
        np.random.seed(self.epoch)
        all_inds = np.empty((0, self.max_num_inds), dtype=int)
        for cat, inds in self.cat2inds.items():
            num_append = self.max_num_inds - len(inds)
            inds_append = (inds * (1 + num_append // len(inds))
                           + inds[:num_append % len(inds)])
            np.random.shuffle(inds_append)
            all_inds = np.vstack([all_inds, inds_append])
        all_inds = all_inds.transpose().flatten().tolist()

        # subsample
        num_samples = len(all_inds) // self.num_replicas
        all_inds = all_inds[
            num_samples * self.rank:num_samples * (self.rank + 1)]
        return iter(all_inds)

    def __len__(self):
        return self.max_num_inds * len(self.cat2inds) // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedSubsetSampler(Sampler):

    def __init__(self, indices):
        num_replicas = get_world_size()
        rank = get_rank()
        self.indices = list(indices)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(np.ceil(len(self.indices) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        np.random.seed(self.epoch)
        np.random.shuffle(self.indices)

        # add extra samples to make it evenly divisible
        self.indices += self.indices[:(self.total_size - len(self.indices))]

        # subsample
        indices = self.indices[self.rank::self.num_replicas]

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class ClassBalancedSubsetSampler(Sampler):

    def __init__(self, dataset, indices):
        self.cat2inds = defaultdict(list)
        for ind in indices:
            self.cat2inds[(dataset.get_cat_ids(ind))].append(ind)
        self.max_num_inds = max(len(_) for _ in self.cat2inds.values())
        self.epoch = 0

    def __iter__(self):
        np.random.seed(self.epoch)
        all_inds = np.empty((0, self.max_num_inds), dtype=int)
        for cat, inds in self.cat2inds.items():
            num_append = self.max_num_inds - len(inds)
            inds_append = (inds * (1 + num_append // len(inds))
                           + inds[:num_append % len(inds)])
            np.random.shuffle(inds_append)
            all_inds = np.vstack([all_inds, inds_append])
        all_inds = all_inds.transpose().flatten().tolist()
        self.epoch += 1
        return iter(all_inds)

    def __len__(self):
        return self.max_num_inds * len(self.cat2inds)
