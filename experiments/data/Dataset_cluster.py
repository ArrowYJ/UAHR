from sklearn.cluster import KMeans, DBSCAN
import torch
import numpy as np
from itertools import chain
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from tqdm import tqdm
import torch.utils.data as data
from experiments import utils
import math
from iirc.utils.utils import print_msg
# This file is used to cluster the latent features of the samples in the dataset.
# The cluster labels are used to sample the same class samples in the same batch.
# Cluster_DataSet = ClusteringDataset(dataset, model, K_Cluster)
# Cluster_Sampler = SameClassSampler(Cluster_DataSet)
# Cluster_BatcherSampler = SameClassBatcherSampler(Cluster_Sampler, batch_size=128, drop_last=True)

class ClusteringDataset(Dataset):
    def __init__(self, dataset, model, K_Cluster, log_book=None):
        self.dataset = dataset
        self.model = model
        self.cluster_labels = []
        self.GT_labels = []
        self.logbook = log_book

        self.cur_task_id = dataset.cur_task_id
        self.cur_tasks = dataset.tasks[self.cur_task_id]
        self.cluster(K_Cluster)

    def cluster(self, K_Cluster, batchsize = 128, num_workers=2):
        # Get latent features from the model of each sample, while add a default cluster label in self.cluster_labels and add the true label in GT_labels
		# get the index of the same class from the GT_labels individually.
		# Cluster the latent features with the same class, and get the cluster_labels which revise the related self.cluster_labels 
        latent_features = []
        train_loader = data.DataLoader(self.dataset, batch_size=batchsize, shuffle=False,  num_workers=num_workers, pin_memory=True, drop_last=False)
        self.model.eval()
        for minibatch in train_loader:
            img = minibatch[0].cuda()
            label_1 = minibatch[1]

            with torch.no_grad():
                feat = self.model(img)
            latent_features.append(feat.squeeze().tolist())
            self.GT_labels.append(label_1)
            self.cluster_labels.append([-1]*len(label_1))
        latent_features = list(chain.from_iterable(latent_features))
        self.GT_labels = list(chain.from_iterable(self.GT_labels))
        self.cluster_labels = list(chain.from_iterable(self.cluster_labels))

        self.GT_labels = np.array(self.GT_labels)
        self.cluster_labels = np.array(self.cluster_labels)
        latent_features = np.array(latent_features)
        label_log_dict = {}

        for label_name in self.cur_tasks:
            # get the index of the self.GT_labels is the same as the label_name
            index = np.where(self.GT_labels == label_name)[0]
            # Cluster the latent features with the same class and get the cluster id for each sample
            class_index_latent_features = latent_features[index]

            kmeans = KMeans(n_clusters=K_Cluster, random_state=None).fit(class_index_latent_features)
            labels = kmeans.labels_
           
            self.cluster_labels[index] = labels
        #utils.log(0, self.cur_task_id, label_log_dict, self.logbook)
		
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _, _  = self.dataset[idx]
        return img, self.GT_labels[idx], self.cluster_labels[idx]

class SameClassSampler(Sampler):
    def __init__(self, data, epoch=-1):
        self.data = data
        self.epoch = epoch

    def __iter__(self):
        indices = []
        g = torch.Generator()
        g.manual_seed(self.epoch)
        for labels_names in self.data.cur_tasks:
            temp_index = np.where(self.data.GT_labels == labels_names)[0]
            # make the index a shuffle
            if self.epoch != -1:
                temp_indices = torch.randperm(len(temp_index), generator=g).tolist()
                index = temp_index[temp_indices]
            else:
                index = np.random.shuffle(temp_index)

            indices.append(index)
        # make the indeices to be a list, and reshape it to be a 1-D array
        indices = list(chain.from_iterable(indices))
        return iter(indices)

    def __len__(self):
        return len(self.data)


class SameClassBatcherSampler(Sampler):
    def __init__(self, sampler, batch_size=128, drop_last=True):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        i = 0
        sampler_list = list(self.sampler)
        for idx in sampler_list:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
            
            if (i < len(sampler_list) - 1 and self.sampler.data.GT_labels[idx] != self.sampler.data.GT_labels[sampler_list[i + 1]]):
                if len(batch) > 0 and not self.drop_last:
                    yield batch
                    batch = []
                else:
                    batch = []
            i += 1
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(list(self.sampler)) // self.batch_size
        else:
            return (len(list(self.sampler)) + self.batch_size - 1) // self.batch_size
class DDPSameclassSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, num_replicas, rank, shuffle=True, batch_size=128, epoch=0):
        super(DDPSameclassSampler, self).__init__(dataset, num_replicas, rank, shuffle)
        self.batch_size = batch_size
        self.epoch = epoch

    def __iter__(self):
        # deterministically shuffle based on epoch
        base_sampler = SameClassSampler(self.dataset, self.epoch)
        indices = list(base_sampler)

        # add extra samples to make it evenly divisible
        self.num_samples = int(len(indices) // self.num_replicas)
        self.total_size = int((len(indices) // self.num_replicas)) * self.num_replicas
        indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        #self.length = len(indices)

        batch = []
        i = 0
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
            if (i < len(indices) - 1 and self.dataset.GT_labels[idx] != self.dataset.GT_labels[indices[i + 1]]):
                if len(batch) > 0:
                    yield batch
                    batch = []
                else:
                    batch = []
            i += 1
        if len(batch) > 0:
            yield batch

    def __len__(self):
        return (len(self.dataset)+self.batch_size-1)//self.batch_size
    
class DDPBaseSampler(Sampler):
    def __init__(self, my_indices, bathch_size=128):
        self.my_indices = my_indices
        self.batch_size = bathch_size

    def __iter__(self):
        batch = []
        i = 0
        sampler_list = self.my_indices
        for idx in sampler_list:
            batch.append(idx)
            i += 1
            if len(batch) == self.batch_size:
                yield batch
                batch = [] 
            if (i == len(sampler_list)):
                if len(batch) > 0:
                    yield batch
                    batch = []
                else:
                    batch = []
        if len(batch) > 0:
            yield batch

    def __len__(self):
        return (len(self.my_indices) + self.batch_size - 1) // self.batch_size
        
class DDPClusteringDataset(Dataset):
    def __init__(self, dataset, model, K_Cluster, log_book=None, rank=0, batchsize=128, world_size=4):
        self.dataset = dataset
        self.model = model
        self.cluster_labels = []
        self.GT_labels = []
        self.logbook = log_book
        
        self.rank = rank
        self.batchsize = batchsize
        self.world_size = world_size
        sample_num = len(self.dataset) // self.world_size
        total_size = sample_num * self.world_size
        self.batch_indices = list(range(self.rank, total_size, self.world_size))

        self.cur_task_id = dataset.cur_task_id
        self.cur_tasks = dataset.tasks[self.cur_task_id]
        self.cluster(K_Cluster)

    def cluster(self, K_Cluster, num_workers=2):
        # Get latent features from the model of each sample, while add a default cluster label in self.cluster_labels and add the true label in GT_labels
		# get the index of the same class from the GT_labels individually.
		# Cluster the latent features with the same class, and get the cluster_labels which revise the related self.cluster_labels 
        
        latent_features = []
        base_sampler = DDPBaseSampler(self.batch_indices, self.batchsize)
        train_loader = data.DataLoader(self.dataset, num_workers=num_workers, pin_memory=True, batch_sampler=base_sampler)
        self.model.eval()
        for minibatch in train_loader:
                img = minibatch[0].cuda()
                label_1 = minibatch[1]
                with torch.no_grad():
                    feat = self.model(img)
                latent_features.append(feat.squeeze().tolist())
                self.GT_labels.append(label_1)
                self.cluster_labels.append([-1]*len(label_1))
        latent_features = list(chain.from_iterable(latent_features))
        self.GT_labels = list(chain.from_iterable(self.GT_labels))
        self.cluster_labels = list(chain.from_iterable(self.cluster_labels))
        
        self.GT_labels = np.array(self.GT_labels)
        self.cluster_labels = np.array(self.cluster_labels)
        latent_features = np.array(latent_features)

        for label_name in self.cur_tasks:
            index = np.where(self.GT_labels == label_name)[0]
            class_index_latent_features = latent_features[index]
            kmeans = KMeans(n_clusters=K_Cluster, random_state=None).fit(class_index_latent_features)
            labels = kmeans.labels_
           
            self.cluster_labels[index] = labels
        #utils.log(0, self.cur_task_id, label_log_dict, self.logbook)
		
    def __len__(self):
        return len(self.batch_indices)

    def __getitem__(self, idx):
        img, _, _  = self.dataset[self.batch_indices[idx]]
        return img, self.GT_labels[idx], self.cluster_labels[idx]

class DDPBaseBatchSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, num_replicas, rank, shuffle=True, batch_size=128, epoch=0):
        super(DDPBaseBatchSampler, self).__init__(dataset, num_replicas, rank, shuffle)
        self.batch_size = batch_size
        self.epoch = epoch

    def __iter__(self):
        # deterministically shuffle based on epoch
        base_sampler = SameClassSampler(self.dataset, self.epoch)
        indices = list(base_sampler)

        batch = []
        i = 0
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
            if (i < len(indices) - 1 and self.dataset.GT_labels[idx] != self.dataset.GT_labels[indices[i + 1]]):
                if len(batch) > 0:
                    yield batch
                    batch = []
                else:
                    batch = []
            i += 1
        if len(batch) > 0:
            yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size





		