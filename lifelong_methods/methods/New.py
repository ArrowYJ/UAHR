from random import sample
from pytz import common_timezones
import torch.nn as nn
from iirc.utils.utils import print_msg
import torch.distributed as dist
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pdb
from PIL import Image
import warnings
from typing import Optional, Union, List, Dict, Callable, Tuple
import torch.utils.data as data
import torch.distributed as dist
from lifelong_methods.utils import transform_labels_names_to_vector
from experiments import utils

from iirc.lifelong_dataset.torch_dataset import Dataset
from iirc.definitions import NO_LABEL_PLACEHOLDER
from lifelong_methods.buffer.buffer import BufferBase
from lifelong_methods.methods.base_method import BaseMethod
from lifelong_methods.utils import SubsetSampler, copy_freeze


class Model(BaseMethod):
    """
    An  implementation of modified version of iCaRL that doesn't use the nearest class mean during inference
    """

    def __init__(self, n_cla_per_tsk: Union[np.ndarray, List[int]], class_names_to_idx: Dict[str, int], config: Dict):
        super(Model, self).__init__(n_cla_per_tsk, class_names_to_idx, config)

        self.old_net = copy_freeze(self.net)
        self.temp = self.temperature
        self.l_divide = config['l_divide']
        self.l_margin = config['l_margin']

        self.m_value = config['m_value']
        self.n_value = config['n_value']
        self.class_value = config['class_value']

    #    self.rbf_dis = config['rbf_dis']
        self.rbf_dis = 1.0
        self.label_relationship = None
        self.label_new_father = None
        # setup losses
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def cal_rbf_single_without_rbf(self, x, y):
        return ((x - y) ** 2).mean().mul(-1)

    def cal_rbf_single(self, x, y):
        return ((x - y) ** 2).mean().div(2 * self.rbf_dis ** 2).mul(-1).exp()
    
    def loss_ranking(self, x, y, offset1, k = 2):
        device = x.device
        possitive_negtive = {'this is the y index':'[[positive_index], [negtive_index]]'}
        for i in range(offset1, len(y[0])):
            positive_index = torch.where(y[:, i] == 1)[0]
            negtive_index = torch.where(y[:, i] == 0)[0]
            possitive_negtive[i] = [positive_index, negtive_index]
        with torch.no_grad():
            dis_near_matrix = self.dis_mm(x.clone())

        ranking_loss = torch.tensor(0.0).to(device)

        for i in range(len(x)):
            c_index = torch.where(y[i, :] == 1)[0][-1]
            if c_index.item() < offset1:
                continue
            try:
                positive_set = possitive_negtive[c_index.item()][0]
                negtive_set = possitive_negtive[c_index.item()][1]
            except:
                pdb.set_trace()
            if len(positive_set) > 1:
                # find the farthest positive sample by torch.topk
                _, index = torch.topk(dis_near_matrix[i, positive_set], k, -1, largest = False)
                farthest_positive = positive_set[index[1]]
                # cal the distance between farthest_positive and x[i]
                dis_farthest_positive = self.cal_rbf_single(x[i], x[farthest_positive])
                if dis_farthest_positive < self.m_value:
                    ranking_loss = ranking_loss - dis_farthest_positive + self.m_value 
            
            if len(negtive_set) > 1:
                # find the nearest negtive sample by torch.topk
                _, index = torch.topk(dis_near_matrix[i, negtive_set], k, -1, largest = True)
                nearest_negtive = negtive_set[index[0]]
                dis_nearest_negtive = self.cal_rbf_single(x[i], x[nearest_negtive])
                if dis_nearest_negtive > self.n_value:
                    ranking_loss = ranking_loss + dis_nearest_negtive - self.n_value
        ranking_loss = ranking_loss / len(x)

        return ranking_loss
        
    def cluster_train(self, x, y, train=True):
        # x is the image, y is the label
        # get the latent feat 
        output, latent_feat = self.forward_net(x)

        cluster_loss = self.loss_ranking(x=latent_feat, y=y, offset1=0)
        no_use_loss = output.mean().mean() - output.mean().mean()
        
        all_loss = cluster_loss + no_use_loss

        if train:
            self.opt.zero_grad()
            all_loss.backward()
            self.opt.step()
            
        return cluster_loss.item()
    

    def loss_divide(self, x, y, offset1, k = 2):
        device = x.device
        cur_pro = torch.tensor([]).to(device)
        new_class_num = 0
        for i in range(offset1, len(y[0])):
            c_index = torch.where(y[:, i] == 1)[0]
            if len(c_index) == 0:
                continue
            new_class_num = new_class_num + 1
            c_x = x[c_index, :]
            c_x = c_x.mean(0)
            cur_pro = torch.cat((cur_pro, c_x), 0)
        
        if new_class_num <= 1:
            return torch.tensor(0.0000001).to(device)

        has_common_father = False
        father_and_son = {-2:[-1,-1]}
        if self.cur_task_id > 0:
            temp_relation = torch.sum(self.label_relationship, 0)
            father_index = torch.where(temp_relation > 1.0)[0]
            if (len(father_index) > 0):
                father_iter = 0
                for index in father_index:
                    son_class = torch.where(self.label_relationship[:, index] == 1)[0]
                    
                    temp_father_proto = torch.tensor([]).to(device)
                    temp_son_num = 0
                    for p in son_class:
                        p_index = torch.where(y[:, offset1 + p] == 1)[0]
                        if len(p_index) == 0:
                            continue
                        p_x = x[p_index, :]
                        temp_father_proto = torch.cat((temp_father_proto, p_x), 0)
                        temp_son_num = temp_son_num + 1
                    if temp_son_num <= 1:
                        continue
                    else:
                        has_common_father = True
                        father_and_son[new_class_num + father_iter] = son_class
                        father_iter = father_iter + 1
                        
                        temp_father_proto = temp_father_proto.view(-1, self.latent_dim)
                        temp_father_proto = temp_father_proto.mean(0)
                        cur_pro = torch.cat((cur_pro, temp_father_proto), 0)
        
        try:
            cur_proto = cur_pro.view(-1, self.latent_dim)
        except:
            pdb.set_trace()
        
        if len(cur_proto) == 2:
            k = 1

        with torch.no_grad():
            dis_near_matrix = self.dis_mm(cur_proto.clone())
        #    dis_near_matrix = self.dis_mm_without_rbf(cur_proto.clone())
        try: 
            _, index = torch.topk(dis_near_matrix, k, -1)
        except:
            pdb.set_trace()
        near_fea_loss = torch.tensor(0.0).to(device)
        for i in range(new_class_num):
            for j in range(k):
                temp_near_loss = self.cal_rbf_single(cur_proto[i, :], cur_proto[index[i][j], :])
                if has_common_father and index[i][j] in father_and_son.keys() and i in father_and_son[index[i][j]]:
                    temp_near_loss = temp_near_loss * 0.01
            #    temp_near_loss = self.cal_rbf_single_without_rbf(cur_proto[i, :], cur_proto[index[i], :])
                near_fea_loss = near_fea_loss + temp_near_loss
        if near_fea_loss == 0:
            return torch.tensor(0.0000001).to(device)
        else:
            return (near_fea_loss / new_class_num)
    
    def dis_mm_without_rbf(self, x):
        left = x.clone()
        right = x.clone()
        diff_temp = left.unsqueeze(1) - right.unsqueeze(0)
        diff = (diff_temp ** 2).mean(2).mul(-1)
        for i in range(len(x)):
            diff[i, i] = 0
        return diff

    def dis_mm(self, x):
        left = x.clone()
        right = x.clone()
        diff_temp = left.unsqueeze(1) - right.unsqueeze(0)
        diff = (diff_temp ** 2).mean(2).div(2 * self.rbf_dis ** 2).mul(-1).exp()
        for i in range(len(x)):
            diff[i, i] = -1
        return diff

    def get_label_relationship(self, cur_old_output, cur_target, offset_1, offset_2, old_latent_feat):
        device = cur_target.device
        label_relationship = torch.zeros([offset_2 - offset_1, offset_2]).to(device)
        with torch.no_grad():
            new_target = cur_target.clone()
            each_session = [0]
            for j in range(self.cur_task_id):
                each_session.append(each_session[j] + self.n_cla_per_tsk[j])
            # find the father
            label_new_father = {}
            for i in range(offset_1, offset_2):
                i_th_sample = torch.where(new_target[:, i] == 1)[0]
                if len(i_th_sample) == 0:
                    continue
                i_th_old_output = cur_old_output[i_th_sample, :]
                i_th_old_latent_feat = old_latent_feat[i_th_sample, :]
            #    before_label = self.get_single_label_relation(i_th_old_output, each_session)
            #    before_label = self.get_single_label_relation_2(i_th_old_output, each_session, i_th_old_latent_feat) 
                before_label = self.get_single_label_relation_fast(i_th_old_output, each_session, i_th_old_latent_feat)
                
                before_label = before_label.type(torch.int64)

                if before_label == -1:
                    label_relationship[i - offset_1, i] = 1
                    label_new_father[i] = 1
                else:
                    label_relationship[i - offset_1, before_label] = 1
                    label_relationship[i - offset_1, i] = 1 
                    label_new_father[i] = 0
        return  label_relationship, label_new_father

    def refine_label(self, cur_old_output, cur_target, offset_1, label_relationship, label_new_father, cur_new_output=None):
        new_target = cur_target.clone()
        is_new_father = [False] * len(cur_old_output)
        for i in range(len(new_target)):
            the_index = torch.where(new_target[i, :] == 1)[0]
            if len(the_index) == 0 and cur_new_output is not None:
                # make the
            #    new_target[i, offset_1:] = cur_new_output[i, offset_1:] - self.l_margin
                continue
            if len(the_index) >= 2 or len(the_index) == 0 :
                continue
                
           # pdb.set_trace()                
            if  the_index >= offset_1 and the_index.item() in label_new_father.keys():
                if label_new_father[the_index.item()] == 1:
                    is_new_father[i] = True
                    _, fake_father = torch.max(cur_old_output[i, :], 0)
                    #TODO: there is a choice------------------------------------------
                    if cur_old_output[i, fake_father] >= 0.5:
                    #    new_target[i, fake_father] = 0.45
                        new_target[i, fake_father] = new_target[i, fake_father] - self.l_margin
                            
                else:
                    son_index = the_index
                    father_index = torch.where(label_relationship[(son_index - offset_1).item(), :] == 1)[0]
                    #TODO: there is a choice------------------------------------------
                    if cur_old_output[i, father_index[0]] < 0.5:
                     #   new_target[i, father_index[0]] = 0.55
                        new_target[i, father_index[0]] = new_target[i, father_index[0]] + self.l_margin
                    
                    try:
                        brother_score = 0.0
                        brother_index = -1
                        father_value = cur_old_output[i, father_index[0]].item()
                        cur_old_output[i, father_index[0]] = 0
                        brother_score, brother_index = torch.max(cur_old_output[i, :], 0)
                        cur_old_output[i, father_index[0]] = father_value
                        if brother_index != -1 and brother_score > 0.5:
                            new_target[i, brother_index] = new_target[i, brother_index] - self.l_margin
                    except:
                        pdb.set_trace()
                    
        return new_target


    def clean_label(self, cur_old_output, cur_target, offset_1, offset_2, old_latent_feat):
        label_relationship, label_new_father = self.get_label_relationship(cur_old_output, cur_target, offset_1, offset_2, old_latent_feat)     
        new_target = self.refine_label(cur_old_output, cur_target, offset_1, label_relationship, label_new_father)  
        return new_target
    
    def get_single_label_relation_fast(self, i_th_old_output, each_session, old_latent_feat):
        device = i_th_old_output.device

        #get the mean and std to judge the close class(seen)
        mean_label = i_th_old_output.mean(0)
        close_label_vec = torch.zeros([self.cur_task_id]).to(device)
        for k in range(self.cur_task_id):
            max_score, max_index = torch.max(mean_label[each_session[k] : each_session[k + 1]], 0)
            if max_score > 0.50:
                close_label_vec[k] = each_session[k] + max_index
            else:
                close_label_vec[k] = -1
        
        # get the std of the old_latent_feat to judge the super-class
        my_label_std = old_latent_feat.std(0).mean()
        close_label_std = torch.zeros([self.cur_task_id]).to(device)
        for k in range(self.cur_task_id):
            if close_label_vec[k] != -1:
                label_sample_index = torch.where(i_th_old_output[:, close_label_vec[k].long()] > 0.5)[0]
                label_sample_index2 = torch.where(i_th_old_output[:, close_label_vec[k].long()] < 0.1)[0]
                if len(label_sample_index) < len(i_th_old_output) / 2 or len(label_sample_index2) > len(i_th_old_output) / 4:
                    close_label_vec[k] = -1
                else:
                    close_label_std[k] = old_latent_feat[label_sample_index, :].std(0).mean()

        # get the distance between current label and close labels. 
        std_dis = torch.abs(close_label_std - my_label_std)
        min_score, min_index = torch.min(std_dis, 0)
        final_label = close_label_vec[min_index]
        return final_label

    def get_single_label_relation_2(self, i_th_old_output, each_session, old_latent_feat):
        device = i_th_old_output.device
        initial_label_matrix = torch.zeros([len(i_th_old_output), self.cur_task_id]).to(device)
        max_label = torch.zeros([self.cur_task_id])
        max_label_count = torch.zeros([self.cur_task_id])

        for j in range(len(i_th_old_output)):
            for k in range(self.cur_task_id):
                max_score, max_index = torch.max(i_th_old_output[j, each_session[k] : each_session[k + 1]], 0)
                if max_score > 0.5:
                    initial_label_matrix[j, k] = each_session[k] + max_index
                else:
                    initial_label_matrix[j, k] = -1
        min_count = 129
        for i in range(self.cur_task_id):
            count_dict = {}
            for j in range(len(i_th_old_output)):
                if initial_label_matrix[j, i].item() in count_dict.keys():
                    count_dict[initial_label_matrix[j, i].item()] = count_dict[initial_label_matrix[j, i].item()] + 1
                else:
                    count_dict[initial_label_matrix[j, i].item()] = 1
            temp_label = 0
            temp_label_count = -1
            for item in count_dict.keys():
                if count_dict[item] > temp_label_count:
                    temp_label = item
                    temp_label_count = count_dict[item] 
            if (temp_label != -1 and temp_label_count < (len(i_th_old_output) / 2)):
                temp_label = -1    
            max_label[i] = temp_label
            max_label_count[i] = temp_label_count
            if max_label[i] != -1 and min_count > temp_label_count:
                min_count = temp_label_count
    #    pdb.set_trace()
        label_std = torch.zeros([len(max_label)]).to(device)
        for label_iter in range(len(max_label)):
            if max_label[label_iter] == -1:
                continue
            else:
                sample_index = torch.where(initial_label_matrix[:, label_iter] == max_label[label_iter])[0]
                equal_index = torch.randperm(len(sample_index))
                c_feat = old_latent_feat[sample_index[equal_index[:min_count]], :]
                label_std[label_iter] = c_feat.std()
        my_label_std = old_latent_feat.std()
        std_dis = torch.abs(label_std - my_label_std)
        min_score, min_index = torch.min(std_dis, 0)
        final_label = max_label[min_index]
        return final_label


    def _load_method_state_dict(self, state_dicts: Dict[str, Dict]) -> None:
        """
        This is where anything model specific needs to be done before the state_dicts are loaded

        Args:
            state_dicts (Dict[str, Dict]): a dictionary with the state dictionaries of this method, the optimizer, the
            scheduler, and the values of the variables whose names are inside the self.method_variables
        """
        pass

    

    def _prepare_model_for_new_task(self, task_data: Dataset, dist_args: Optional[dict] = None, num_workers: int = 2, device: torch.device = None, log_book=None,
                                    **kwargs) -> None:
        """
        A method specific function that takes place before the starting epoch of each new task (runs from the
        prepare_model_for_task function).
        It copies the old network and freezes it's gradients.
        """ 
        distributed = dist_args is not None
        if distributed:
            self.old_net = copy_freeze(self.net.module)
        else:
            self.old_net = copy_freeze(self.net)
        self.old_net.eval()
        cur_task_id = self.cur_task_id
        if  cur_task_id == 0:
            return 

        batch_size = 128
        base_epoch = 20
        thresh_hold = 0.65
        
        if distributed:
            gpu = dist_args["gpu"]
            rank = dist_args["rank"]
        else:
            gpu = None
            rank = 0
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(task_data, num_replicas=dist_args["world_size"], rank=rank)
        else:
            train_sampler = None
        dataloader = data.DataLoader(task_data, batch_size=batch_size, shuffle=(train_sampler is None),num_workers=num_workers, pin_memory=True, sampler=train_sampler)
    
        offset1, offset2 = self._compute_offsets(self.cur_task_id)
        ori_label_relationship = torch.zeros([offset2 - offset1, offset2]).to(device)

        epoch = -1
        label_log = {}
        
        with torch.no_grad():
            for epoch in range(0, base_epoch):
                if distributed:
                    train_sampler.set_epoch(epoch)
                class_names_to_idx = self.class_names_to_idx
                num_seen_classes = len(self.seen_classes)
                minibatch_i = 0
                for minibatch in dataloader:  
                    labels_names = list(zip(minibatch[1], minibatch[2]))
                    labels = transform_labels_names_to_vector(
                        labels_names, num_seen_classes, class_names_to_idx
                    )
                    if gpu is None:
                        images = minibatch[0].to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                    else:
                        images = minibatch[0].to(torch.device(f"cuda:{gpu}"), non_blocking=True)
                        labels = labels.to(torch.device(f"cuda:{gpu}"), non_blocking=True)

                    y = labels.clone()
                    distill_model_output, old_latent_feat = self.old_net(images)
                    distill_model_output = distill_model_output.detach()
                    old_latent_feat = old_latent_feat.detach()
                    distill_model_output = torch.sigmoid(distill_model_output / self.temp)
                    y[:, :offset1] = distill_model_output[:, :offset1]
                    label_relationship, _ = self.get_label_relationship(distill_model_output[:, :offset1], y[:, :offset2], offset1, offset2, old_latent_feat)
                    ori_label_relationship = ori_label_relationship + label_relationship
                   # if cur_task_id == 2:
                    #    print(f'ori_label_relationship:{ori_label_relationship}')
                     #   print(f'label_relationship:{label_relationship}')
                      #  pdb.set_trace()
        #print(ori_label_relationship)
        max_number = ori_label_relationship.max()
        thresh_number = max_number * thresh_hold
        final_label_relationship = torch.zeros([offset2 - offset1, offset2]).to(device)
        label_new_father = {}
        for cross in range(offset2 - offset1):
            not_judged_father = True
            for row in range(offset2):
                if ori_label_relationship[cross, row] >= thresh_number:
                    final_label_relationship[cross, row] = 1
                    if row < offset1 and not_judged_father:
                        label_new_father[cross + offset1] = 0
                        not_judged_father = False
                    if row >= offset1 and not_judged_father:
                        label_new_father[cross + offset1] = 1
                        not_judged_father = False
        print(final_label_relationship)
        #print(label_new_father)
        #pdb.set_trace()
        self.label_relationship = final_label_relationship
        self.label_new_father = label_new_father
        if rank == 0:
    #        label_log["label_relation"] = f"{self.label_relationship}"
    #        utils.log_task(self.cur_task_id, label_log, log_book)
            if distributed:
                dist.barrier()
        else:
            if distributed:
                dist.barrier()
                    
    def _preprocess_target(self, x: torch.Tensor, y: torch.Tensor, cur_new_output=None) -> torch.Tensor:
        """Replaces the labels on the older classes with the distillation targets produced by the old network"""
        offset1, offset2 = self._compute_offsets(self.cur_task_id)
        y = y.clone()
        if self.cur_task_id > 0:
            distill_model_output, old_latent_feat = self.old_net(x)
            distill_model_output = distill_model_output.detach()
            old_latent_feat = old_latent_feat.detach()
        #    distill_model_output = torch.sigmoid(distill_model_output / self.temperature)
            distill_model_output = torch.sigmoid(distill_model_output / self.temp)
            y[:, :offset1] = distill_model_output[:, :offset1]

            y = self.refine_label(distill_model_output[:, :offset1], y[:, :offset2], offset1, self.label_relationship, self.label_new_father, cur_new_output=cur_new_output)
        return y

    def get_cls_loss(self, output, target, offset1, in_buffer, latent_feat=None, x=None):
        if self.cur_task_id < 1 or in_buffer is None:
            loss_1 = self.bce(output / self.temp, target)
            return loss_1

        for i in range(len(output)):
            if not in_buffer[i]:
                c_index = torch.where(target[i, :] == 1)[0]
                c_x = c_index[-1]
                if self.label_new_father[int(c_x)] == 0:   # subclass
                    father_index = torch.where(self.label_relationship[c_x - offset1, :] == 1)[0]
                    father_index = father_index[0]
                    temp_output_father = torch.sigmoid(output[i, father_index] / self.temp)
                    temp_c_x = torch.sigmoid(output[i, c_x] / self.temp)
                    if abs(temp_output_father - temp_c_x) > self.class_value * 0.9:
                        output[i, c_x] = (output[i, c_x] + output[i, father_index]) / 2 - ((abs(temp_output_father - temp_c_x) - self.class_value * 0.9).exp() - 1.0)
              #      if abs(output[i, mix_before] - output[i, c_x]) > self.class_value:  # doing----------------------
              #          output[i, c_x] = output[i, c_x] - abs(output[i, mix_before] - output[i, c_x]).mul(-1).exp() * 0.3
              #      output[i, c_x] = (output[i, c_x] + output[i, father_index]) / 2
                
                else:    # superclass
                    mix_before = torch.max(output[i, :offset1], 0)[1]
                    temp_output_mix = torch.sigmoid(output[i, mix_before] / self.temp)
                    temp_c_x = torch.sigmoid(output[i, c_x] / self.temp)
                    if abs(temp_output_mix - temp_c_x) < self.class_value * 1.1:
                        output[i, c_x] = output[i, c_x] - ((self.class_value * 1.1 - abs(temp_output_mix - temp_c_x)).exp() - 1.0)
                
        loss_1 = self.bce(output / self.temp, target)
        # A distilled from the old net fc. then the output to be the same 
        if latent_feat is not None:
            distill_model_output, _ = self.old_net(x)

            distill_model_output = distill_model_output.detach()
            distill_model_output = torch.sigmoid(distill_model_output / self.temp)
            output_old_new_latent = self.old_net.model.output_layer(latent_feat)
            loss_1_distill = self.bce(output_old_new_latent / self.temp, distill_model_output)
        
        return loss_1 + loss_1_distill * 0.1

    def observe(self, x: torch.Tensor, y: torch.Tensor, in_buffer: Optional[torch.Tensor] = None,
                train: bool = True) -> Tuple[torch.Tensor, float]:
        """
        The method used for training and validation, returns a tensor of model predictions and the loss
        This function needs to be defined in the inheriting method class

        Args:
            x (torch.Tensor): The batch of images
            y (torch.Tensor): A 2-d batch indicator tensor of shape (number of samples x number of classes)
            in_buffer (Optional[torch.Tensor]): A 1-d boolean tensor which indicates which sample is from the buffer.
            train (bool): Whether this is training or validation/test

        Returns:
            Tuple[torch.Tensor, float]:
            predictions (torch.Tensor) : a 2-d float tensor of the model predictions of shape (number of samples x number of classes)
            loss (float): the value of the loss
        """

        offset_1, offset_2 = self._compute_offsets(self.cur_task_id)
        output, latent_feat = self.forward_net(x)
        output = output[:, :offset_2]
        target = self._preprocess_target(x, y,  cur_new_output=output)
        assert target.shape[1] == offset_2
        
        loss_1 = self.get_cls_loss(output, target, offset_1, in_buffer, latent_feat=latent_feat, x=x)
        loss_divide = self.loss_divide(latent_feat, y, offset_1)
    #    loss_1 = self.bce(output / self.temperature, target)
        
        loss_2 =  (loss_1.item() * loss_divide) / (loss_divide.item() * self.l_divide + 1)  
     #   loss_ranking_value = self.loss_ranking(latent_feat, y, offset_1)
    #    loss_3 = loss_ranking_value * (loss_1.item() / (loss_ranking_value.item() * self.l_divide + 1))
        loss = loss_1  + loss_2 #+ loss_3  #+ loss_2# + loss_3 #+ loss_2

        if train:
            self.opt.zero_grad()
            loss.backward()
        #    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
            self.opt.step()

        predictions = output.ge(0.0)
        return predictions, loss.item()

    def forward(self, x: torch.Tensor, return_output=False) -> torch.Tensor:
        """
        The method used during inference, returns a tensor of model predictions

        Args:
            x (torch.Tensor): The batch of images

        Returns:
            torch.Tensor: a 2-d float tensor of the model predictions of shape (number of samples x number of classes)
        """
        num_seen_classes = len(self.seen_classes)
        output_ori, _ = self.forward_net(x)
        output_ori = output_ori[:, :num_seen_classes]
        # make the output to between 0 and 1 by sigmoid
        if self.cur_task_id >= 1:
            output = torch.sigmoid(output_ori / self.temp)
            # make the value is between the (max value - self.class_value) and the max value to be positive and the others to be negative
            output = output - torch.max(output, 1, keepdim=True)[0] + self.class_value
        else:
            output = output_ori
        predictions = output.ge(0.0)
        if return_output:
            return predictions, output_ori
        else:
            return predictions

    def _consolidate_epoch_knowledge(self, **kwargs) -> None:
        """
        A method specific function that takes place after training on each epoch (runs from the
        consolidate_epoch_knowledge function)
        """
        pass

    def consolidate_task_knowledge(self, **kwargs) -> None:
        """Takes place after training on each task"""
        pass


class Buffer(BufferBase):
    def __init__(self,
                 config: Dict,
                 buffer_dir: Optional[str] = None,
                 map_size: int = 1e9,
                 essential_transforms_fn: Optional[Callable[[Image.Image], torch.Tensor]] = None,
                 augmentation_transforms_fn: Optional[Callable[[Image.Image], torch.Tensor]] = None):
        super(Buffer, self).__init__(config, buffer_dir, map_size, essential_transforms_fn, augmentation_transforms_fn)

    def _reduce_exemplar_set(self, **kwargs) -> None:
        """remove extra exemplars from the buffer"""
        for label in self.seen_classes:
            if len(self.mem_class_x[label]) > self.n_mems_per_cla:
                n = len(self.mem_class_x[label]) - self.n_mems_per_cla
                self.remove_samples(label, n)

    def _construct_exemplar_set(self, task_data: Dataset, dist_args: Optional[dict] = None,
                                model: torch.nn.Module = None, batch_size=1, **kwargs):
        """
        Update the buffer with the new task samples using herding

        Args:
            task_data (Dataset): The new task data
            dist_args (Optional[Dict]): a dictionary of the distributed processing values in case of multiple gpu (ex:
            rank of the device) (default: None)
            model (BaseMethod): The current method object to calculate the latent variables
            batch_size (int): The minibatch size
        """
        distributed = dist_args is not None
        if distributed:
            device = torch.device(f"cuda:{dist_args['gpu']}")
            rank = dist_args['rank']
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            rank = 0
        new_class_labels = task_data.cur_task
        model.eval()

        with task_data.disable_augmentations(): # disable augmentations then enable them (if they were already enabled)
            with torch.no_grad():
                for class_label in new_class_labels:
                    class_data_indices = task_data.get_image_indices_by_cla(class_label, self.max_mems_pool_size)
                    if distributed:
                        device = torch.device(f"cuda:{dist_args['gpu']}")
                        class_data_indices_to_broadcast = torch.from_numpy(class_data_indices).to(device)
                        dist.broadcast(class_data_indices_to_broadcast, 0)
                        
                        class_data_indices = class_data_indices_to_broadcast.cpu().numpy()
                    sampler = SubsetSampler(class_data_indices)
                    class_loader = DataLoader(task_data, batch_size=batch_size, sampler=sampler)
                    latent_vectors = []
                    for minibatch in class_loader:
                        images = minibatch[0].to(device)
                        output, out_latent = model.forward_net(images)
                        out_latent = out_latent.detach()
                        out_latent = F.normalize(out_latent, p=2, dim=-1)
                        latent_vectors.append(out_latent)
                    latent_vectors = torch.cat(latent_vectors, dim=0)
                    class_mean = torch.mean(latent_vectors, dim=0)

                    chosen_exemplars_ind = []
                    exemplars_mean = torch.zeros_like(class_mean)
                    while len(chosen_exemplars_ind) < min(self.n_mems_per_cla, len(class_data_indices)):
                        potential_exemplars_mean = (exemplars_mean.unsqueeze(0) * len(chosen_exemplars_ind) + latent_vectors) \
                                                   / (len(chosen_exemplars_ind) + 1)
                        distance = (class_mean.unsqueeze(0) - potential_exemplars_mean).norm(dim=-1)
                        shuffled_index = torch.argmin(distance).item()
                        exemplars_mean = potential_exemplars_mean[shuffled_index, :].clone()
                        exemplar_index = class_data_indices[shuffled_index]
                        chosen_exemplars_ind.append(exemplar_index)
                        latent_vectors[shuffled_index, :] = float("inf")

                    for image_index in chosen_exemplars_ind:
                        minibatch = task_data.get_item(image_index)
                        image = minibatch[0]
                        label1 = minibatch[1]
                        label2 = minibatch[2]
                        if label2 != NO_LABEL_PLACEHOLDER:
                            warnings.warn(f"Sample is being added to the buffer with labels {label1} and {label2}")
                        self.add_sample(class_label, image, (label1, label2), rank=rank)
