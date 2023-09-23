from cProfile import label
import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler
import numpy as np
import random
import pdb

class EachClassSampler(Sampler):
    def __init__(self, data):
        self.data = data
        self.cur_task_id = data.cur_task_id
        self.cur_tasks = data.tasks[self.cur_task_id]
        self.num_class = len(self.cur_tasks)
        label_array = np.array(list([label1, label2] for _, label1, label2 in self.data))[:, 0]
        
        self.each_label_index = []
        max_length = 0
        for label_name in self.cur_tasks:
            label_index = np.where(label_array == label_name)[0]
            random.shuffle(label_index)
            label_length = len(label_index)
            if max_length < label_length:
                max_length = label_length
            self.each_label_index.append(label_index)

        self.max_class_length = max_length
        self.length = max_length * self.num_class


    def __iter__(self):
        indices = []
        each_label_iter = [0] * self.num_class

        for i in range(self.max_class_length):
            for nth_class in range(self.num_class):
                cur_index = each_label_iter[nth_class]
                if cur_index < len(self.each_label_index[nth_class]):
                    indices.append(self.each_label_index[nth_class][cur_index])
                    each_label_iter[nth_class] = cur_index + 1
                else:
                    indices.append(self.each_label_index[nth_class][0])
                    each_label_iter[nth_class] = 1
        # indices type() ---------doing
        return iter(indices)

    def __len__(self):
        return self.length
		
class EachClassBatchSampler:
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.num_class = self.sampler.num_class
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.real_size = (self.batch_size // self.num_class) * self.num_class

    def __iter__(self):
        batch = []
        i = 0
        sampler_list = list(self.sampler)
        for idx in sampler_list:
            batch.append(idx)
            if len(batch) == self.real_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.real_size
        else:
            return (len(self.sampler) + self.real_size - 1) // self.real_size


class EachClassSampler_Buffer(Sampler):
    def __init__(self, data):
        self.buffer_batch = 5
        self.data = data
        self.cur_task_id = data.task_data.cur_task_id
        self.cur_tasks = data.task_data.tasks[self.cur_task_id]
        self.num_class = len(self.cur_tasks)

        buffer_data_index = self.data._get_buffer_index_sampling_array() 
        if len(buffer_data_index) == 0:
            self.buffer_data_index = []
            self.buffer_batch = 0
        else:
            random.shuffle(buffer_data_index)
            self.buffer_data_index = buffer_data_index

        self.each_label_index = []
        max_length = 0
        for label_name in self.cur_tasks:
            label_index = self.data.task_data.get_image_indices_by_cla(label_name)
            label_length = len(label_index)
            if max_length < label_length:
                max_length = label_length
            self.each_label_index.append(label_index)

        self.max_class_length = max_length
        self.length = max_length * (self.num_class + self.buffer_batch)


    def __iter__(self):
        indices = []
        '''
        if len(self.buffer_data_index) == 0:
            each_label_iter = [0] * self.num_class
            for i in range(self.max_class_length):
                for nth_class in range(self.num_class):
                    cur_index = each_label_iter[nth_class]
                    if cur_index < len(self.each_label_index[nth_class]):
                        indices.append(self.each_label_index[nth_class][cur_index])
                        each_label_iter[nth_class] = cur_index + 1
                    else:
                        indices.append(self.each_label_index[nth_class][0])
                        each_label_iter[nth_class] = 1
        else:
        '''
        each_label_iter = [0] * (self.num_class + 1)
        for i in range(self.max_class_length):
            for nth_class in range(self.num_class):
                cur_index = each_label_iter[nth_class]
                if cur_index < len(self.each_label_index[nth_class]):
                    indices.append(self.each_label_index[nth_class][cur_index])
                    each_label_iter[nth_class] = cur_index + 1
                else:
                    indices.append(self.each_label_index[nth_class][0])
                    each_label_iter[nth_class] = 1
            for nth_buffer in range(self.buffer_batch):
                cur_index = each_label_iter[-1]
                if cur_index < len(self.buffer_data_index):
                    indices.append(self.buffer_data_index[cur_index])
                    each_label_iter[-1] = cur_index + 1
                else:
                    indices.append(self.buffer_data_index[0])
                    each_label_iter[-1] = 1
        return iter(indices)

    def __len__(self):
        return self.length
'''
class EachClassSampler_Buffer(Sampler):
    def __init__(self, data):
        self.buffer_batch = 5
        self.data = data
        self.cur_task_id = data.task_data.cur_task_id
        self.cur_tasks = data.task_data.tasks[self.cur_task_id]
        self.num_class = len(self.cur_tasks)
        label_array = np.array(list([label1, label2, buffer] for _, label1, label2, buffer in self.data))

        buffer_data_index = np.where(label_array[:, 2] == True)[0]
        if len(buffer_data_index) == 0:
            self.buffer_data_index = []
            self.buffer_batch = 0
        else:
            random.shuffle(buffer_data_index)
            self.buffer_data_index = buffer_data_index

        self.each_label_index = []
        max_length = 0
        for label_name in self.cur_tasks:
            label_index = np.where(label_array[:, 0] == label_name)[0]
            random.shuffle(label_index)
            label_length = len(label_index)
            if max_length < label_length:
                max_length = label_length
            self.each_label_index.append(label_index)

        self.max_class_length = max_length
        self.length = max_length * (self.num_class + self.buffer_batch)


    def __iter__(self):
        indices = []
    
        each_label_iter = [0] * (self.num_class + 1)
        for i in range(self.max_class_length):
            for nth_class in range(self.num_class):
                cur_index = each_label_iter[nth_class]
                if cur_index < len(self.each_label_index[nth_class]):
                    indices.append(self.each_label_index[nth_class][cur_index])
                    each_label_iter[nth_class] = cur_index + 1
                else:
                    indices.append(self.each_label_index[nth_class][0])
                    each_label_iter[nth_class] = 1
            for nth_buffer in range(self.buffer_batch):
                cur_index = each_label_iter[-1]
                if cur_index < len(self.buffer_data_index):
                    indices.append(self.buffer_data_index[cur_index])
                    each_label_iter[-1] = cur_index + 1
                else:
                    indices.append(self.buffer_data_index[0])
                    each_label_iter[-1] = 1
        return iter(indices)

    def __len__(self):
        return self.length
'''

class Support_ClassSampler_Buffer(Sampler):
    def __init__(self, data):
        self.data = data
        self.cur_task_id = data.task_data.cur_task_id
        self.cur_tasks = data.task_data.tasks[self.cur_task_id]
        self.num_class = len(self.cur_tasks)

        self.buffer_data_index = []
        self.buffer_batch = 0
 
        self.each_label_index = []
        max_length = 0
        for label_name in self.cur_tasks:
            label_index = self.data.task_data.get_image_indices_by_cla(label_name)
            label_length = len(label_index)
            if max_length < label_length:
                max_length = label_length
            self.each_label_index.append(label_index)

        self.max_class_length = max_length
        self.length = max_length * (self.num_class + self.buffer_batch)


    def __iter__(self):
        indices = []
        each_label_iter = [0] * (self.num_class + 1)
        for i in range(self.max_class_length):
            for nth_class in range(self.num_class):
                cur_index = each_label_iter[nth_class]
                if cur_index < len(self.each_label_index[nth_class]):
                    indices.append(self.each_label_index[nth_class][cur_index])
                    each_label_iter[nth_class] = cur_index + 1
                else:
                    indices.append(self.each_label_index[nth_class][0])
                    each_label_iter[nth_class] = 1
        return iter(indices)

    def __len__(self):
        return self.length
'''
class Support_ClassSampler_Buffer(Sampler):
    def __init__(self, data):
        self.data = data
        self.cur_task_id = data.task_data.cur_task_id
        self.cur_tasks = data.task_data.tasks[self.cur_task_id]
        self.num_class = len(self.cur_tasks)
        label_array = np.array(list([label1, label2, buffer] for _, label1, label2, buffer in self.data))

        self.buffer_data_index = []
        self.buffer_batch = 0
 

        self.each_label_index = []
        max_length = 0
        for label_name in self.cur_tasks:
            label_index = np.where(label_array[:, 0] == label_name)[0]
            random.shuffle(label_index)
            label_length = len(label_index)
            if max_length < label_length:
                max_length = label_length
            self.each_label_index.append(label_index)

        self.max_class_length = max_length
        self.length = max_length * (self.num_class + self.buffer_batch)


    def __iter__(self):
        indices = []
        each_label_iter = [0] * (self.num_class + 1)
        for i in range(self.max_class_length):
            for nth_class in range(self.num_class):
                cur_index = each_label_iter[nth_class]
                if cur_index < len(self.each_label_index[nth_class]):
                    indices.append(self.each_label_index[nth_class][cur_index])
                    each_label_iter[nth_class] = cur_index + 1
                else:
                    indices.append(self.each_label_index[nth_class][0])
                    each_label_iter[nth_class] = 1
        return iter(indices)

    def __len__(self):
        return self.length
'''

class SingleClassSampler_Buffer(Sampler):
    def __init__(self, data):
        self.data = data
        self.cur_task_id = data.task_data.cur_task_id
        self.cur_tasks = data.task_data.tasks[self.cur_task_id]
        self.num_class = len(self.cur_tasks)
        label_array = np.array(list([label1, label2, buffer] for _, label1, label2, buffer in self.data))

        buffer_data_index = np.where(label_array[:, 2] == True)[0]
        self.buffer_batch = 0
        if len(buffer_data_index) == 0:
            self.buffer_data_index = []
        else:
            random.shuffle(buffer_data_index)
            self.buffer_data_index = buffer_data_index

        self.each_label_index = []
        max_length = 0
        for label_name in self.cur_tasks:
            label_index = np.where(label_array[:, 0] == label_name)[0]
            random.shuffle(label_index)
            label_length = len(label_index)
            max_length = max_length +  label_length
            self.each_label_index.append(label_index)

        self.length = max_length + len(self.buffer_data_index)
  #      pdb.set_trace()


    def __iter__(self):
        indices = []
        for i in range(self.num_class):
            for p_index in range(len(self.each_label_index[i])):
                indices.append(self.each_label_index[i][p_index])
        for nth_buffer in range(len(self.buffer_data_index)):
            indices.append(self.buffer_data_index[nth_buffer])
        return iter(indices)

    def __len__(self):
        return self.length


class EachClassBatchSampler_Buffer():
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.buffer_batch = self.sampler.buffer_batch
        self.num_class = self.sampler.num_class
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.real_size = (self.batch_size // (self.num_class + self.buffer_batch)) * (self.num_class + self.buffer_batch)

    def __iter__(self):
        batch = []
        i = 0
        sampler_list = list(self.sampler)
        for idx in sampler_list:
            batch.append(idx)
            if len(batch) == self.real_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.real_size
        else:
            return (len(self.sampler) + self.real_size - 1) // self.real_size

