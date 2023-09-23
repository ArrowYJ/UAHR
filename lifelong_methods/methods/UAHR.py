from random import sample
from pytz import common_timezones
import torch.nn as nn
import torch.distributed as dist
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pdb
from PIL import Image
import warnings
from typing import Optional, Union, List, Dict, Callable, Tuple

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
        self.rbf_dis = config['rbf_dis']

        # setup losses
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def cal_rbf_single_without_rbf(self, x, y):
        return ((x - y) ** 2).mean().mul(-1)

    def cal_rbf_single(self, x, y):
        return ((x - y) ** 2).mean().div(2 * self.rbf_dis ** 2).mul(-1).exp()

    def loss_divide(self, x, y, offset1):
        device = x.device
        cur_pro = torch.tensor([]).to(device)
        for i in range(offset1, len(y[0])):
            c_index = torch.where(y[:, i] == 1)[0]
            if len(c_index) == 0:
                continue
            c_x = x[c_index, :]
            c_x = c_x.mean(0)
            cur_pro = torch.cat((cur_pro, c_x), 0)
        try:
            cur_proto = cur_pro.view(-1, self.latent_dim)
        except:
            pdb.set_trace()

        new_class_num = len(cur_proto)
        if new_class_num == 0:
            return torch.tensor(0.00001).to(device)
        with torch.no_grad():
            dis_near_matrix = self.dis_mm(cur_proto.clone())
        #    dis_near_matrix = self.dis_mm_without_rbf(cur_proto.clone())
        try: 
            _, index = torch.max(dis_near_matrix, 1)
        except:
            pdb.set_trace()
        near_fea_loss = torch.tensor(0.0).to(device)
        for i in range(new_class_num):
            temp_near_loss = self.cal_rbf_single(cur_proto[i, :], cur_proto[index[i], :])
        #    temp_near_loss = self.cal_rbf_single_without_rbf(cur_proto[i, :], cur_proto[index[i], :])
            near_fea_loss = near_fea_loss + temp_near_loss
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
            diff[i, i] = 0
        return diff

    def get_label_relationship(self, cur_old_output, cur_target, offset_1, offset_2, old_latent_feat):
        device = cur_target.device
        is_new_father = [False] * len(cur_old_output)
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
            # refine label        
            for i in range(len(new_target)):
                the_index = torch.where(new_target[i, :] == 1)[0]
                if len(the_index) == 0 or len(the_index) == 2:
                    continue
               # pdb.set_trace()
                
                if  the_index >= offset_1 and the_index.item() in label_new_father.keys():
                    if label_new_father[the_index.item()] == 1:
                        is_new_father[i] = True
                        _, fake_father = torch.max(cur_old_output[i, :], 0)
                        if cur_old_output[i, fake_father] >= 0.5:
                        #    new_target[i, fake_father] = 0.45
                            new_target[i, fake_father] = new_target[i, fake_father] - self.l_margin
                            
                    else:
                        son_index = the_index
                        father_index = torch.where(label_relationship[(son_index - offset_1).item(), :] == 1)[0]
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
                            if brother_score > 0.5 and brother_index != -1:
                                new_target[i, brother_index] = new_target[i, brother_index] - self.l_margin
                        except:
                            pdb.set_trace()
                            
                
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
                if len(label_sample_index) < len(i_th_old_output) / 2:
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

    def get_single_label_relation(self, i_th_old_output, each_session):
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
        label_mean = torch.zeros([len(max_label)]).to(device)
        my_label_mean = torch.zeros([len(max_label)]).to(device)
        my_label_std = torch.zeros([len(max_label)]).to(device)
        for label_iter in range(len(max_label)):
            if max_label[label_iter] == -1:
                my_label_mean[label_iter] = 9999999999
                my_label_std[label_iter] = 99999999999
                continue
            else:
                sample_index = torch.where(initial_label_matrix[:, label_iter] == max_label[label_iter])[0]
                equal_index = torch.randperm(len(sample_index))
                try:
                    c_feat = i_th_old_output[sample_index[equal_index[:min_count]].type(torch.int64).cuda(), max_label[label_iter].type(torch.int64).cuda()]
                except:
                    pdb.set_trace()
                label_mean[label_iter] = c_feat.mean()
        #        label_std[label_iter] = c_feat.std()
        #        all_c_feat = i_th_old_output[:, max_label[label_iter].type(torch.int64).cuda()]
        #        my_label_mean[label_iter] = all_c_feat.mean()
     #           my_label_std[label_iter] = all_c_feat.std()      
    #    std_dis = torch.abs(label_std - my_label_std)
    #    mean_dis = torch.abs(label_mean - my_label_mean)
        #if self.cur_task_id == 3:
        #    print(f'max_label:{max_label}\n max_label_count:{max_label_count}\n label_mean:{label_mean}\n label_std:{label_std} \n std_dis:{std_dis} \n mean_dis:{mean_dis}\n')  
        #    pdb.set_trace()
    #    min_score, min_index = torch.min(std_dis, 0)
    #    min_score, min_index = torch.min(mean_dis, 0)
        '''
        final_label = torch.tensor(-1.0)
        max_count = 0
        label_mean_temp = -1
        for i in range(len(max_label)):
            if max_label[i] != -1:
                if max_label_count[i] > max_count:
                    final_label = max_label[i]
                    max_count = max_label_count[i]
                    label_mean_temp = label_mean[i]
                elif max_label_count[i] == max_count and label_mean[i] > label_mean_temp:
                    final_label = max_label[i]
                    max_count = max_label_count[i]
                    label_mean_temp = label_mean[i]
        '''               
    #    final_label = max_label[min_index]
        final_label = torch.tensor(-1)
        temp_mean = 0
        for i in range(len(max_label)):
            if max_label[i] != -1:
                if label_mean[i] > temp_mean:
                    final_label = max_label[i]
                    temp_mean = label_mean[i]
        
        return final_label

    def _load_method_state_dict(self, state_dicts: Dict[str, Dict]) -> None:
        """
        This is where anything model specific needs to be done before the state_dicts are loaded

        Args:
            state_dicts (Dict[str, Dict]): a dictionary with the state dictionaries of this method, the optimizer, the
            scheduler, and the values of the variables whose names are inside the self.method_variables
        """
        pass

    

    def _prepare_model_for_new_task(self, **kwargs) -> None:
        """
        A method specific function that takes place before the starting epoch of each new task (runs from the
        prepare_model_for_task function).
        It copies the old network and freezes it's gradients.
        """
        self.old_net = copy_freeze(self.net)
        self.old_net.eval()

    def _preprocess_target(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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

            y = self.get_label_relationship(distill_model_output[:, :offset1], y[:, :offset2], offset1, offset2, old_latent_feat)
        return y

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
        target = self._preprocess_target(x, y)
        assert target.shape[1] == offset_2


        output, latent_feat = self.forward_net(x)
        output = output[:, :offset_2]
        loss_divide = self.loss_divide(latent_feat, y, offset_1)
    #    loss_1 = self.bce(output / self.temperature, target)
        loss_1 = self.bce(output / self.temp, target)
        loss_2 =  (loss_1.item() / (loss_divide.item() * self.l_divide + 1)) * loss_divide 
        loss = loss_1  + loss_2

        if train:
            self.opt.zero_grad()
            loss.backward()
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
        output, _ = self.forward_net(x)
        output = output[:, :num_seen_classes]
        predictions = output.ge(0.0)
        if return_output:
            return predictions, output
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
