from cProfile import label
from itertools import count
import torch.nn as nn
import torch.distributed as dist
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import warnings
from typing import Optional, Union, List, Dict, Callable, Tuple
import math
import pdb

from iirc.lifelong_dataset.torch_dataset import Dataset
from iirc.definitions import NO_LABEL_PLACEHOLDER
from iirc.utils.utils import print_msg
from lifelong_methods.buffer.buffer import BufferBase
from lifelong_methods.methods.base_method import BaseMethod
from lifelong_methods.utils import SubsetSampler, copy_freeze, l_distance
from lifelong_methods.models import cosine_linear
from lifelong_methods.models.resnetcifar import ResNetCIFAR
from lifelong_methods.models.resnet import ResNet


class Model(BaseMethod):
    """
    An  implementation of LUCIR from
        Saihui Hou, Xinyu Pan, Chen Change Loy, Zilei Wang, and Dahua Lin.
        Learning a Unified Classifier Incrementally via Rebalancing.
        CVPR, 2019.
    """

    def __init__(self, n_cla_per_tsk: Union[np.ndarray, List[int]], class_names_to_idx: Dict[str, int], config: Dict):
        super(Model, self).__init__(n_cla_per_tsk, class_names_to_idx, config)

        self.sigma = True
        device = next(self.net.parameters()).device
   #     pdb.set_trace()
        if config['n_layers'] not in [20, 32, 44, 56, 110]:
            self.net.model.fc = cosine_linear.CosineLinear(in_features=self.latent_dim, out_features=self.latent_dim, sigma=self.sigma).to(device)
        else:
            self.net.model.output_layer = cosine_linear.CosineLinear(in_features=self.latent_dim, out_features=self.latent_dim, sigma=self.sigma).to(device)                                                  
   #     self.net.model.output_layer = nn.Linear(self.latent_dim, self.latent_dim)

        # duq part-------------------------------------------------------------  first task class num
        self.register_buffer(
            "m", torch.normal(torch.zeros(n_cla_per_tsk[0], self.latent_dim), 0.05)
        )
        self.gama = config['gama']
        

        self.reset_optimizer_and_scheduler()
        self.old_net = copy_freeze(self.net)  # type: Union[ResNet, ResNetCIFAR]
        self.register_buffer(
            "old_m", torch.normal(torch.zeros(n_cla_per_tsk[0], self.latent_dim), 0.05)
        )

        self.batch_size = config["batch_size"]

        self.lambda_base = config["lucir_lambda"]
        self.lambda_cur = self.lambda_base
        self.K = 2
        self.margin_1 = config["lucir_margin_1"]
        self.margin_2 = config["lucir_margin_2"]

        self.l_distance_m = config['l_distance_m']
        self.l_distance_divide = config['l_distance_divide']
        self.l_gradient_penalty = config['l_gradient_penalty']
        self.l_cls = config['l_cls']
        self.iter_count = config['iter_count']
        self.ge_border = config['ge_border']

        # setup losses
        # self.loss_classification = nn.CrossEntropyLoss(reduction="mean")
        self.loss_classification = nn.BCELoss(reduction="mean")
        self.loss_distill = nn.CosineEmbeddingLoss(reduction="mean")
        '''
        # several losses to allow for the use of different margins
        self.loss_mr_1 = nn.MarginRankingLoss(margin=self.margin_1, reduction="mean")
        self.loss_mr_2 = nn.MarginRankingLoss(margin=self.margin_2, reduction="mean")
        '''
        self.method_variables.extend(["lambda_base", "lambda_cur", "K", "margin_1", "margin_2", "sigma", "l_gradient_penalty", "l_distance_divide", "l_distance_m", "l_cls", "iter_count"])

    # get the cls score------------------------------------------------------done
    def dis_rbf(self, z):
        diff = z.unsqueeze(1) - self.m.unsqueeze(0)
        output = (diff ** 2).mean(2).div(2 * self.sigma ** 2).mul(-1).exp()
        return output

    def dis_old_rbf(self, z):
        self.old_m = self.old_m.cuda()
       # pdb.set_trace()
        embeddings = self.old_m
        diff = z.unsqueeze(1) - embeddings.unsqueeze(0)
        output = (diff ** 2).mean(2).div(2 * self.sigma ** 2).mul(-1).exp()
        return output

    def sam_rbf(self, z):
        embeddings = self.m
        fen_mu = torch.matmul((z ** 2).sum(1), (embeddings ** 2).sum(1).T)
        fen_zi = torch.matmul(z, embeddings.T)
        diff = torch.abs(torch.acos(torch.div(fen_zi, fen_mu))).mul(-1).exp()
        return diff

    #get the new embeddings-------------------------------------------------done
    def update_embeddings_from_image(self, x, y, offset1):
        count_start = offset1
        self.N = self.gama * self.N + (1 - self.gama) * len(torch.where(y[:, count_start] == 1)[0])
        for i in range(count_start, len(y[0])):
            y_to_c = torch.where(y[:, i] == 1)[0]
            z, _ = self.forward_net(x[y_to_c])
            embedding_sum = torch.sum(z, 0)
            self.m[i] = self.gama * self.m[i] + (1 - self.gama) * embedding_sum

    def dis_whole_mm(self, x, y, ):
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        diff = (diff ** 2).mean(2).div(2 * self.sigma ** 2).mul(-1).exp()
        return diff

    def update_embeddings_from_fea(self, x_fea, y, offset1):
        count_start = offset1
        ori_m = self.m.clone()
        new_m = self.m.clone()
        for i in range(count_start, len(y[0])):
            y_to_c = torch.where(y[:, i] == 1)[0]
            if len(y_to_c) == 0:
                continue
            z = x_fea[y_to_c].clone()
            new_m[i] = self.gama * ori_m[i] + (1 - self.gama) * z.mean(0)
   #     pdb.set_trace()
        temp_m = new_m.clone()
        cur_fea = temp_m[count_start:]
      #  std_fea = ((cur_fea - cur_fea.mean(0))**2).sum(1).sum(0).sqrt().mul(-1).exp()

        dis_near_matrix = self.dis_mm(cur_fea.clone())
        _, index = torch.max(dis_near_matrix, 1)
        near_fea_loss = torch.tensor(0.0).cuda()
        for i in range(len(index)):
            temp_near_loss = ((cur_fea[i,:] - cur_fea[index[i], :])**2).mean().div(2 * self.sigma ** 2).mul(-1).exp()
            near_fea_loss = near_fea_loss + temp_near_loss

        loss_father_and_son = torch.tensor(0.0).cuda()
        for i in range(len(y)):
            y_index = torch.where(y[i, :] == 1)[0]
            if len(y_index)==2 and (y_index[0] - offset1) * (y_index[1] - offset1) <= 0:
                father_index = y_index[0]
                son_index = y_index[1]
                if y_index[0] > y_index[1]:
                    father_index = y_index[1]
                    son_index = y_index[0]
                temp_loss_father_and_son = ((cur_fea[son_index - offset1, :] - temp_m[father_index, :])**2).mean().div(2 * self.sigma ** 2).mul(-1).exp()    
                loss_father_and_son = loss_father_and_son - temp_loss_father_and_son

        all_loss =  near_fea_loss / len(index)  + loss_father_and_son  #+ std_fea 
        return all_loss, new_m.clone().detach()

    # get the loss of gradient------------------------------------------------done
    def calc_gradients_input(self, x, y_pred):
        gradients = torch.autograd.grad(
            outputs=y_pred,
            inputs=x.requires_grad_(),
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
       #     allow_unused=True
        )[0]

        gradients = gradients.flatten(start_dim=1)

        return gradients

    # get the penalty of the gradient----------------------------------done
    def calc_gradient_penalty(self, x, y_pred):
        gradients = self.calc_gradients_input(x, y_pred)

        # L2 norm
        grad_norm = gradients.norm(2, dim=1)

        # Two sided penalty
        gradient_penalty = ((grad_norm - 1) ** 2).mean()

        return gradient_penalty
    
    def dis_mm(self, x):
        left = x.clone()
        right = x.clone()
        diff_temp = left.unsqueeze(1) - right.unsqueeze(0)
        diff = (diff_temp ** 2).mean(2).div(2 * self.sigma ** 2).mul(-1).exp()
        for i in range(len(x)):
            diff[i, i] = 0
        return diff

    def distanced_fea(self, x_fea, y, offset1):
        count_start = offset1
        temp_fea = torch.normal(torch.zeros(len(y[0]), self.latent_dim), 0.05)

        #inner_cluster_loss = torch.tensor(0.0).cuda()
        for i in range(count_start, len(y[0])):
            y_to_c = torch.where(y[:, i] == 1)[0]
            temp_fea[i, :] = x_fea[y_to_c].mean(0)
         #   cluster_fea = x_fea[y_to_c].clone()
         #   inner_cluster_loss = inner_cluster_loss + ((cluster_fea - cluster_fea.mean(0))**2).sum(1).sum(0).sqrt().exp()

        cur_fea = temp_fea[count_start:]
        new_cur_fea = cur_fea.clone()
      #  std_fea = ((cur_fea - cur_fea.mean(0))**2).sum(1).sum(0).sqrt().mul(-1).exp()

        dis_matrix = self.dis_mm(new_cur_fea.clone())
        _, index = torch.max(dis_matrix, 1)
        near_fea_loss = torch.tensor(0.0).cuda()
        for i in range(len(index)):
            temp_loss = ((new_cur_fea[i, :] - new_cur_fea[index[i]])**2).mean().div(2 * self.sigma **2).mul(-1).exp()
            near_fea_loss = near_fea_loss +  temp_loss
        divide_loss = near_fea_loss  / len(index)

        all_loss = divide_loss # + inner_cluster_loss + std_fea  
        return all_loss

    def _load_method_state_dict(self, state_dicts: Dict[str, Dict]) -> None:
        """
        This is where anything model specific needs to be done before the state_dicts are loaded.
        This method replaces the output layer of the vanilla resnet with the cosine layer, and change the trainable
        parameters.

        Args:
            state_dicts (Dict[str, Dict]): a dictionary with the state dictionaries of this method, the optimizer, the
            scheduler, and the values of the variables whose names are inside the self.method_variables
        """
        '''
        assert "method_variables" in state_dicts.keys()
        method_variables = state_dicts['method_variables']
        cur_task_id = method_variables["cur_task_id"]
        n_cla_per_tsk = method_variables["n_cla_per_tsk"]
        num_old_classes = int(sum(n_cla_per_tsk[: cur_task_id]))
        num_new_classes = n_cla_per_tsk[cur_task_id]
        device = next(self.net.parameters()).device
        '''
        '''
        if cur_task_id > 0:
            self.net.model.output_layer = cosine_linear.SplitCosineLinear(in_features=self.latent_dim,
                                                                          out_features1=num_old_classes,
                                                                          out_features2=num_new_classes,)
           
            trainable_parameters = [param for name, param in self.net.named_parameters() if
                                    "output_layer.fc1" not in name]
            self.reset_optimizer_and_scheduler(trainable_parameters)
            if cur_task_id > 1:
                out_features1 = int(sum(n_cla_per_tsk[: cur_task_id - 1]))
                out_features2 = n_cla_per_tsk[cur_task_id - 1]
                self.old_net.model.output_layer = cosine_linear.SplitCosineLinear(in_features=self.latent_dim,
                                                                                  out_features1=out_features1,
                                                                                  out_features2=out_features2,
                                                                                  sigma=self.sigma).to(device)
        '''
    def _prepare_model_for_new_task(self, task_data: Dataset, dist_args: Optional[dict] = None,
                                    **kwargs) -> None:
        """
        A method specific function that takes place before the starting epoch of each new task (runs from the
        prepare_model_for_task function).
        It copies the old network and freezes it's gradients.
        It also extends the output layer, imprints weights for those extended nodes, and change the trainable parameters

        Args:
            task_data (Dataset): The new task dataset
            dist_args (Optional[Dict]): a dictionary of the distributed processing values in case of multiple gpu (ex:
            rank of the device) (default: None)
        """

        self.old_net = copy_freeze(self.net)
        self.old_net.eval()

        self.old_m = self.m.clone().detach()

        cur_task_id = self.cur_task_id
        num_old_classes = int(sum(self.n_cla_per_tsk[: cur_task_id]))
        num_new_classes = self.n_cla_per_tsk[cur_task_id]
        device = next(self.net.parameters()).device
        
        # Extend last layer
        if cur_task_id > 0:
            new_m = torch.normal(torch.zeros(num_old_classes + num_new_classes, self.latent_dim), 0.05)
            new_m[ :num_old_classes, :] = self.m[:num_old_classes, :]
            new_m.cuda()
            self.m = new_m
            self.m.cuda()
            print_msg(f"Lambda for less forget is set to {self.lambda_cur}")
        elif cur_task_id != 0:
            raise ValueError("task id cannot be negative")

    def cls_loss(self, x, y):
        loss_right = torch.tensor(0.0)
        loss_wrong = torch.tensor(0.0)
        num_wrong = torch.tensor(0)
        num_right = torch.tensor(0)
        for i in range(len(y)):
            for j in range(len(y[i])):
                if y[i, j] == 1:
                    loss_right = loss_right - torch.log(x[i, j])
                    num_right = num_right + 1
                else:
                    loss_wrong = loss_wrong - torch.log(1 - x[i, j])
                    num_wrong = num_wrong + 1
        return (loss_right / num_right) + (loss_wrong / num_wrong)

    def observe(self, x: torch.Tensor, y: torch.Tensor, in_buffer: Optional[torch.Tensor] = None,
                train: bool = True, epoch_cur=0) -> Tuple[torch.Tensor, float]:
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
        
        #--------------------------------above is super parameter
        device = x.device
        num_seen_classes = len(self.seen_classes)
        offset_1, offset_2 = self._compute_offsets(self.cur_task_id)
        target = y.clone()
        assert y.shape[1] == offset_2 == num_seen_classes
        x.requires_grad_(True)

        #temp_output = self.forward_net(x)
        temp_output, latent_feat = self.forward_net(x)
   #     gp = self.calc_gradient_penalty(x, temp_output)
        output = self.dis_rbf(temp_output.clone())
        assert output.shape[1] == num_seen_classes
     #   pdb.set_trace()
        if self.cur_task_id == 0:
            loss_1 = self.cls_loss(output[:, offset_1:], target[:, offset_1:]) * self.l_cls
     #       loss_1 = self.loss_classification(output[:, offset_1:], target[:, offset_1:]) * self.l_cls
    #        if train:
    #            gp = self.calc_gradient_penalty(x, output)
    #            loss_1 = loss_1 +  self.l_gradient_penalty * gp
      #      pdb.set_trace()
            if epoch_cur % self.iter_count == self.iter_count - 1:
                    loss_2 = self.distanced_fea(temp_output.clone(), target, offset_1)
                    loss_2_temp, new_m = self.update_embeddings_from_fea(temp_output.clone(), target.clone(), offset_1)
                    loss_2 = (loss_2 + loss_2_temp * self.l_distance_m) *  self.l_distance_divide
            else:
                with torch.no_grad():
                    loss_2_temp, new_m = self.update_embeddings_from_fea(temp_output.clone(), target.clone(), offset_1)
                    loss_2 = torch.zeros_like(loss_1)

        if self.cur_task_id > 0:
            output = self.dis_rbf(temp_output)
            assert output.shape[1] == num_seen_classes
            self.old_net.eval()
           
            old_temp_output, old_latent_feat = self.old_net(x)
            old_output = self.dis_old_rbf(old_temp_output.clone())
            old_output = old_output.detach()
            old_latent_feat = old_latent_feat.detach()
            old_temp_output = old_temp_output.detach()

            new_target = target.clone()
            for i in range(len(old_output)):
                for j in range(len(old_output[0])):
                    if old_output[i, j] == 1:
                        new_target[i, j] = 1
                        break
            loss_1 = self.cls_loss(output, new_target)
       #     gp = self.calc_gradient_penalty(x, output)
       #     loss_1 += l_gradient_penalty * gp

            is_all_buffer = False
            if in_buffer is not None:
                cur_data_index = torch.where(in_buffer == False)[0]
                if (len(cur_data_index) == 0):
                    is_all_buffer = True
                else:
                    cur_output = output[cur_data_index, :].clone()
                    cur_target = new_target[cur_data_index, :].clone()
                    cur_temp_output = temp_output[cur_data_index, :].clone()
            else:
                cur_output = output.clone()
                cur_target = new_target
                cur_temp_output = temp_output
            if is_all_buffer:
                self.lambda_cur = 5.0
            else:
                if epoch_cur % self.iter_count == self.iter_count - 1:
                        loss_2 = self.distanced_fea(cur_temp_output.clone(), cur_target, offset_1)
                        loss_2_temp, new_m = self.update_embeddings_from_fea(cur_temp_output.clone(), cur_target.clone(), offset_1)
                        loss_2 = (loss_2 + loss_2_temp * self.l_distance_m) *  self.l_distance_divide
                else:
                    with torch.no_grad():
                        loss_2_temp, new_m = self.update_embeddings_from_fea(cur_temp_output.clone(), cur_target.clone(), offset_1)
                    loss_2 = torch.zeros_like(loss_1)

            #---------------------strong and weak distill--------------------------------------------------------doing
            loss_3 = self.loss_distill(latent_feat.clone(), old_latent_feat, torch.ones(x.shape[0]).to(device)) * self.lambda_cur                                      
            loss_3 += self.loss_distill(temp_output.clone(), old_temp_output, torch.ones(x.shape[0]).to(device)) * self.lambda_cur
   

            loss_4 = torch.zeros_like(loss_1)
            '''
            if in_buffer is not None and torch.sum(in_buffer) > 0:
                buffer_target = y[in_buffer]
                buffer_output = output[in_buffer]
                loss_4  = self.loss_classification(buffer_output[:, :offset_1], buffer_target[:, :offset_1])
            '''
            '''
                sigma = self.net.model.output_layer.sigma
                if sigma is not None:
                    buffer_output /= sigma.data
                assert buffer_target.shape[0] == torch.sum(in_buffer)  ####
                # target_scores = buffer_output.masked_select(buffer_target).view(-1, 1).repeat(1, self.K).reshape(-1)
                target_scores = buffer_output.masked_select(buffer_target).reshape(-1)
                new_classes_scores = buffer_output[:, offset_1:]
                topk_scores, _ = new_classes_scores.topk(self.K, dim=-1)
                topk_scores_1 = topk_scores[:, 0].reshape(-1)
                topk_scores_2 = topk_scores[:, 1:].reshape(-1)
                loss_4 = self.loss_mr_1(target_scores.reshape(-1), topk_scores_1, torch.ones_like(topk_scores_1))
                loss_4 += self.loss_mr_2(target_scores.view(-1, 1).repeat(1, self.K - 1).reshape(-1), topk_scores_2,
                                         torch.ones_like(topk_scores_2)) * (self.K - 1)
                loss_4 /= self.K
                '''
        else:
            loss_3 = torch.zeros_like(loss_1)
            loss_4 = torch.zeros_like(loss_1)

        if train:
            loss = loss_1 # + loss_2 # + loss_3 + loss_4
            self.opt.zero_grad()
       #     torch.autograd.set_detect_anomaly(True)
            loss.backward(retain_graph=True)
            self.opt.step()
            x.requires_grad_(False)
            self.m = new_m
        else:
            loss = loss_1
        predictions = output.ge(self.ge_border)

        return predictions, loss.item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The method used during inference, returns a tensor of model predictions

        Args:
            x (torch.Tensor): The batch of images

        Returns:
            torch.Tensor: a 2-d float tensor of the model predictions of shape (number of samples x number of classes)
        """
        num_seen_classes = len(self.seen_classes)
        temp_output, latent_fea = self.forward_net(x)
        output = self.dis_rbf(temp_output)
        assert output.shape[1] == num_seen_classes
        predictions = output.ge(self.ge_border)
      #  pdb.set_trace()
        return predictions, temp_output, latent_fea

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
                                model: torch.nn.Module = None, batch_size=1, **kwargs) -> None:
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

        print_msg(f"Adding buffer samples")  ####
        with task_data.disable_augmentations():  # disable augmentations then enable them (if they were already enabled)
            with torch.no_grad():
                for class_label in new_class_labels:
                    class_data_indices = task_data.get_image_indices_by_cla(class_label, self.max_mems_pool_size)
                    if distributed:
                        device = torch.device(f"cuda:{dist_args['gpu']}")
                        class_data_indices_to_broadcast = torch.from_numpy(class_data_indices).to(device)
                        torch.distributed.broadcast(class_data_indices_to_broadcast, 0)
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
                        potential_exemplars_mean = (exemplars_mean.unsqueeze(0) * len(
                            chosen_exemplars_ind) + latent_vectors) \
                                                   / (len(chosen_exemplars_ind) + 1)
                        distance = (class_mean.unsqueeze(0) - potential_exemplars_mean).norm(dim=-1)
                        shuffled_index = torch.argmin(distance).item()
                        exemplars_mean = potential_exemplars_mean[shuffled_index, :].clone()
                        exemplar_index = class_data_indices[shuffled_index]
                        chosen_exemplars_ind.append(exemplar_index)
                        latent_vectors[shuffled_index, :] = float("inf")

                    for image_index in chosen_exemplars_ind:
                        image, label1, label2 = task_data.get_item(image_index)
                        if label2 != NO_LABEL_PLACEHOLDER:
                            warnings.warn(f"Sample is being added to the buffer with labels {label1} and {label2}")
                        self.add_sample(class_label, image, (label1, label2), rank=rank)
