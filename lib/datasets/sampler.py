from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler
from mmcv.runner import get_dist_info


class CombinedSampler(Sampler):

    def __init__(self, samplers):
        self.samplers = samplers
        self.epoch = 0

    def __iter__(self):
        inds = [list(iter(sampler)) for sampler in self.samplers]
        min_num_inds = min(len(_) for _ in inds)
        inds = [ind[:min_num_inds] for ind in inds]
        inds = np.vstack(inds)
        inds = inds.transpose().flatten().tolist()
        return iter(inds)

    def __len__(self):
        return (len(self.samplers)
                * min(len(sampler) for sampler in self.samplers))

    def set_epoch(self, epoch):
        self.epoch = epoch
        for sampler in self.samplers:
            sampler.set_epoch(epoch)


class DistributedClassBalancedSubsetSampler(Sampler):

    def __init__(self, dataset, indices):
        self.rank, self.num_replicas = get_dist_info()
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

    def __init__(self, indices, shuffle=False):
        rank, num_replicas = get_dist_info()
        self.indices = list(indices)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(np.ceil(len(self.indices) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        np.random.seed(self.epoch)
        indices = self.indices[:] # copy
        if self.shuffle:
            np.random.shuffle(indices)

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]

        # subsample
        indices = indices[self.rank::self.num_replicas]

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
        return iter(all_inds)

    def __len__(self):
        return self.max_num_inds * len(self.cat2inds)

    def set_epoch(self, epoch):
        self.epoch = epoch


class ReversedSubsetSampler(Sampler):

    def __init__(self, dataset, indices):
        self.cat2inds = defaultdict(list)
        for ind in indices:
            self.cat2inds[(dataset.get_cat_ids(ind))].append(ind)
        reci_probs = {cat: num_total * 1. / len(inds)
                 for cat, inds in self.cat2inds.items()}
        sum_reci_probs = sum(_ for _ in reci_probs.values())
        self.nums_reversed = {
            cat: int(len(indices) * 10 * reci_prob / sum_reci_probs)
            for cat, reci_prob in reci_probs.items()}
        self.num_total = int(np.ceil(len(indices) / self.num_replicas)
                             * self.num_replicas)
        self.epoch = 0

    def __iter__(self):
        np.random.seed(self.epoch)
        all_inds = []
        for cat, num_reversed in self.nums_reversed.items():
            inds = self.cat2inds[cat]
            inds = np.array(inds)[
                np.random.randint(0, len(inds), (num_reversed,))]
            all_inds += inds.tolist()
        np.random.shuffle(all_inds)
        all_inds = all_inds[:self.num_total]
        return iter(all_inds)

    def __len__(self):
        return self.num_total

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedReversedSubsetSampler(Sampler):

    def __init__(self, dataset, indices):
        self.rank, self.num_replicas = get_dist_info()
        self.cat2inds = defaultdict(list)
        for ind in indices:
            self.cat2inds[(dataset.get_cat_ids(ind))].append(ind)
        reci_probs = {cat: num_total * 1. / len(inds)
                 for cat, inds in self.cat2inds.items()}
        sum_reci_probs = sum(_ for _ in reci_probs.values())
        self.nums_reversed = {
            cat: int(len(indices) * 10 * reci_prob / sum_reci_probs)
            for cat, reci_prob in reci_probs.items()}
        self.num_total = int(np.ceil(len(indices) / self.num_replicas)
                             * self.num_replicas)
        self.epoch = 0

    def __iter__(self):
        np.random.seed(self.epoch)
        all_inds = []
        for cat, num_reversed in self.nums_reversed.items():
            inds = self.cat2inds[cat]
            rand_inds = np.array(inds)[
                np.random.randint(0, len(inds), (num_reversed,))]
            all_inds += rand_inds.tolist()
        np.random.shuffle(all_inds)
        all_inds = all_inds[:self.num_total]
        # inds on each device
        all_inds = all_inds[self.rank:self.num_total:self.num_replicas]
        return iter(all_inds)

    def __len__(self):
        return self.num_total // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch
