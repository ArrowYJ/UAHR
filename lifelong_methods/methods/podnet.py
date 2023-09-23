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
from sklearn.cluster import KMeans
from math import ceil, floor, sqrt, factorial

from iirc.lifelong_dataset.torch_dataset import Dataset
from iirc.definitions import NO_LABEL_PLACEHOLDER
from iirc.utils.utils import print_msg
from lifelong_methods.buffer.buffer import BufferBase
from lifelong_methods.methods.base_method import BaseMethod
from lifelong_methods.utils import SubsetSampler, copy_freeze
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
        self.num_proxy = config['num_proxy']
        self.sigma = True
        device = next(self.net.parameters()).device
        self.net.model.output_layer = cosine_linear.CosineLinearProxy(in_features=self.latent_dim,
                                                                 out_features=n_cla_per_tsk[0],
                                                                 num_proxy=self.num_proxy,
                                                                 sigma=self.sigma).to(device)

        self.reset_optimizer_and_scheduler()
        self.old_net = copy_freeze(self.net)  # type: Union[ResNet, ResNetCIFAR]

        self.batch_size = config["batch_size"]

        self.lambda_base = config["lucir_lambda"]
        self.lambda_cur = self.lambda_base
        self.K = 2
        self.distill_factorize = config["distill_factorize"]
        self.margin_2 = config["lucir_margin_2"]
        self.num_segments = config["num_segments"]

        # setup losses
        # self.loss_classification = nn.CrossEntropyLoss(reduction="mean")
        self.loss_classification = nn.BCEWithLogitsLoss(reduction="mean")
        self.loss_distill = nn.CosineEmbeddingLoss(reduction="mean")
        # several losses to allow for the use of different margins

        self.method_variables.extend(["lambda_base", "lambda_cur", "sigma"])

    def _load_method_state_dict(self, state_dicts: Dict[str, Dict]) -> None:
        """
        This is where anything model specific needs to be done before the state_dicts are loaded.
        This method replaces the output layer of the vanilla resnet with the cosine layer, and change the trainable
        parameters.

        Args:
            state_dicts (Dict[str, Dict]): a dictionary with the state dictionaries of this method, the optimizer, the
            scheduler, and the values of the variables whose names are inside the self.method_variables
        """
        assert "method_variables" in state_dicts.keys()
        method_variables = state_dicts['method_variables']
        cur_task_id = method_variables["cur_task_id"]
        n_cla_per_tsk = method_variables["n_cla_per_tsk"]
        num_old_classes = int(sum(n_cla_per_tsk[: cur_task_id]))
        num_new_classes = n_cla_per_tsk[cur_task_id]
        device = next(self.net.parameters()).device
        if cur_task_id > 0:
            self.net.model.output_layer = cosine_linear.SplitCosineLinearProxy(in_features=self.latent_dim,
                                                                          out_features1=num_old_classes,
                                                                          out_features2=num_new_classes,
                                                                          num_proxy=self.num_proxy,
                                                                          sigma=self.sigma).to(device)
            trainable_parameters = [param for name, param in self.net.named_parameters() if
                                    "output_layer.fc1" not in name]
            self.reset_optimizer_and_scheduler(trainable_parameters)
            if cur_task_id > 1:
                out_features1 = int(sum(n_cla_per_tsk[: cur_task_id - 1]))
                out_features2 = n_cla_per_tsk[cur_task_id - 1]
                self.old_net.model.output_layer = cosine_linear.SplitCosineLinearProxy(in_features=self.latent_dim,
                                                                                  out_features1=out_features1,
                                                                                  out_features2=out_features2,
                                                                                  num_proxy=self.num_proxy,
                                                                                  sigma=self.sigma).to(device)

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

        cur_task_id = self.cur_task_id
        num_old_classes = int(sum(self.n_cla_per_tsk[: cur_task_id]))
        num_new_classes = self.n_cla_per_tsk[cur_task_id]
        device = next(self.net.parameters()).device

        # Extend last layer
        if cur_task_id > 0:
            output_layer = cosine_linear.SplitCosineLinearProxy(in_features=self.latent_dim,
                                                           out_features1=num_old_classes,
                                                           out_features2=num_new_classes,
                                                           num_proxy=self.num_proxy,
                                                           sigma=self.sigma).to(device)
            if cur_task_id == 1:
                output_layer.fc1.weight.data = self.net.model.output_layer.weight.data
            else:
                out_features1 = self.net.model.output_layer.fc1.out_features
                output_layer.fc1.weight.data[:out_features1 * self.num_proxy] = self.net.model.output_layer.fc1.weight.data
                output_layer.fc1.weight.data[self.num_proxy * out_features1:] = self.net.model.output_layer.fc2.weight.data
            output_layer.sigma.data = self.net.model.output_layer.sigma.data
            self.net.model.output_layer = output_layer
            self.lambda_cur = self.lambda_base * math.sqrt(num_old_classes * 1.0 / num_new_classes)
            print_msg(f"Lambda for less forget is set to {self.lambda_cur}")
        elif cur_task_id != 0:
            raise ValueError("task id cannot be negative")

        
        # Imprint weights
        with task_data.disable_augmentations():
            if cur_task_id > 0:
                print_msg("Imprinting weights")
           #     self.net = self._imprint_weights(task_data, self.net, dist_args)
        
        # Fix parameters of FC1 for less forget and reset optimizer/scheduler
        if cur_task_id > 0:
            trainable_parameters = [param for name, param in self.net.named_parameters() if
                                    "output_layer.fc1" not in name]
        else:
            trainable_parameters = self.net.parameters()
        self.reset_optimizer_and_scheduler(trainable_parameters)
        

    def _imprint_weights(self, task_data: Dataset, model: Union[ResNet, ResNetCIFAR],
                         dist_args: Optional[dict] = None) -> Union[ResNet, ResNetCIFAR]:
        distributed = dist_args is not None
        if distributed:
            device = torch.device(f"cuda:{dist_args['gpu']}")
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        class_names = task_data.cur_task
        class_names_2_idx = self.class_names_to_idx
        model.eval()
        num_old_classes = model.model.output_layer.fc1.out_features
        old_weights_norm = model.model.output_layer.fc1.weight.data.norm(dim=1, keepdim=True)
        average_old_weights_norm = torch.mean(old_weights_norm, dim=0)
        new_weights = torch.zeros_like(model.model.output_layer.fc2.weight.data)
        for cla in class_names:
            cla_id = class_names_2_idx[cla]
            if cla_id < num_old_classes:
                continue
            num_samples = 1000
            class_indices = task_data.get_image_indices_by_cla(cla, num_samples=num_samples, shuffle=False)
            if distributed:  # make sure all the gpus use the same random indices
                class_data_indices_to_broadcast = torch.from_numpy(class_indices).to(device)
                torch.distributed.broadcast(class_data_indices_to_broadcast, 0)
                class_indices = class_data_indices_to_broadcast.cpu().numpy()
            sampler = SubsetSampler(class_indices)
            class_loader = DataLoader(task_data, batch_size=self.batch_size, sampler=sampler)
            normalized_latent_feat = []
            with torch.no_grad():
                for minibatch in class_loader:
                    inputs = minibatch[0].to(device)
                    output, latent_features, _ = model(inputs)
                    latent_features = latent_features.detach()
                    latent_features = F.normalize(latent_features, p=2, dim=-1)
                    normalized_latent_feat.append(latent_features)
                normalized_latent_feat = torch.cat(normalized_latent_feat, dim=0)
                clusterizer = KMeans(n_clusters=self.num_proxy)
                clusterizer.fit(normalized_latent_feat.cpu().numpy())
                new_weight_count = 0
                for center in clusterizer.cluster_centers_:
                    try:
                        new_weights[(cla_id - num_old_classes) * self.num_proxy + new_weight_count] = torch.tensor(center).cuda() * average_old_weights_norm
                    except:
                        import pdb
                        pdb.set_trace()
                    new_weight_count = new_weight_count + 1

        model.model.output_layer.fc2.weight.data = new_weights
        return model

    def scale_loss(self, f1,f2,scale):
        num_segments = f1.size()[1]
        num_channels = f1.size()[-1]

        f1 = f1.view(-1,scale,num_segments//scale,num_channels) # (B, S, N/S, C)
        f2 = f2.view(-1,scale,num_segments//scale,num_channels)
        f1 = f1.mean(2) # (B, S, C)
        f2 = f2.mean(2)
        f1 = f1.view(-1,num_channels) # (BxS,C)
        f2 = f2.view(-1,num_channels)
        loss = nn.CosineEmbeddingLoss()(f1,f2.clone().detach(), torch.ones(f1.shape[0]).cuda()).cuda()
        return loss

    def lf_dist(self, feat,feat_old, m_res=False): 
        if m_res:
            loss_1 = self.scale_loss(feat, feat_old, 1)
            loss_2 = self.scale_loss(feat, feat_old, 2)
            loss_4 = self.scale_loss(feat, feat_old, 4)
            loss_8 = self.scale_loss(feat, feat_old, 8)
            loss_dist = loss_1 + loss_2 + loss_4 + loss_8
        else:
        #    feat = feat.mean(1)
        #    feat_old = feat_old.mean(1)
            try:
                loss_dist = nn.CosineEmbeddingLoss()(feat,feat_old.clone().detach(),torch.ones(feat.shape[0]).cuda()).cuda()
            except:
                import pdb
                print(feat.shape)
                print(feat_old.shape)
                pdb.set_trace()
        return loss_dist

    def factorize(self, feat, dim):
        """
        dim = 2(T),3(H),4(W)
        """
        B,C,T,H,W = feat.shape
        '''
        if dim == 2:
            feat = feat.sum(dim=(3,4)) # B,C,T
        elif dim == 3:
            feat = feat.sum(dim=(2,4)) # B,C,H
        elif dim == 4:
            feat = feat.sum(dim=(2,3)) # B,C,W
        '''
        feat = feat.sum(dim=dim) # B,C,d1,d2
        if not isinstance(dim,tuple):
            feat = feat/(feat.norm(dim=(2,3),keepdim=True) + 1e-8)
        else:
            feat = feat/(feat.norm(dim=-1,keepdim=True) + 1e-8)
        feat = feat.view(B,-1)
        return feat

    def feat_dist(self, fmap, fmap_old, factor=None):
        factorize = self.distill_factorize
        num_layers = len(fmap)
        loss_dist = torch.tensor(0.).cuda(non_blocking=True)
        num_segments = self.num_segments
        for i in range(num_layers):
            f1 = fmap[i]
            f1 = f1.view(-1, num_segments, f1.size()[1], f1.size()[2], f1.size()[3]) # (B,T,C,H,W)
            f1 = f1.permute(0,2,1,3,4) # (B,C,T,H,W)
            f2 = fmap_old[i]
            f2 = f2.view(-1, num_segments, f2.size()[1], f2.size()[2], f2.size()[3])
            f2 = f2.permute(0, 2, 1, 3, 4)
            f1 = f1.pow(2)
            f2 = f2.pow(2)
            assert (f1.shape == f2.shape)
            B,C,T,H,W = f1.shape

            if factorize=='T-S':
                f_cur_t = self.factorize(f1,dim=2) # (B,C*H*W)
                f_old_t = self.factorize(f2,dim=2)
                f_cur_s = self.factorize(f1,dim=(3,4)) # (B,C*T)
                f_old_s = self.factorize(f2,dim=(3,4))
                f_cur = torch.cat([f_cur_t,f_cur_s],dim=-1)
                f_old = torch.cat([f_old_t,f_old_s],dim=-1)
                f_cur = F.normalize(f_cur, dim=1, p=2)
                f_old = F.normalize(f_old, dim=1, p=2)
                loss_i = torch.mean(torch.frobenius_norm(f_cur-f_old.clone().detach(),dim=-1))

            elif factorize=='T-GAP':
                f1 = self.factorize(f1,dim=(3,4)) # (B,C*T)
                f2 = self.factorize(f2,dim=(3,4))
                f1 = F.normalize(f1, dim=1, p=2)
                f2 = F.normalize(f2, dim=1, p=2)
                if factor is not None: # Ours
                    factor_i = factor[i].permute(1,0)
                    factor_i = factor[i].reshape([1,-1])
                    loss_i = torch.mean(factor_i * torch.abs(f1-f2.clone().detach()))
                else:
                    loss_i = torch.mean(torch.frobenius_norm(f1-f2.clone().detach(),dim=-1))
                loss_i = loss_i/sqrt(T)

            elif factorize=='T-POD':
                f1_H = f1.sum(3) # B, C, T, W
                f1_W = f1.sum(4) # B, C, T, H
                f2_H = f2.sum(3)
                f2_W = f2.sum(4)
                f1 = torch.cat([f1_H,f1_W],dim=-1) # B, C, T, H+W
                f2 = torch.cat([f2_H,f2_W],dim=-1)
                f1 = F.normalize(f1, dim=-1, p=2)
                f2 = F.normalize(f2, dim=-1, p=2)
                f1 = f1.view(-1,C*T,f1.size()[3])
                f2 = f2.view(-1,C*T,f2.size()[3])
                if factor is not None: # Ours
                    factor_i = factor[i].permute(1,0)
                    factor_i = factor[i].reshape([1,-1])
                    loss_i = torch.mean(factor_i * torch.frobenius_norm(f1-f2.clone().detach(),dim=-1))
                else:
                    loss_i = torch.mean(torch.frobenius_norm(f1-f2.clone().detach(),dim=-1))
                loss_i = loss_i/sqrt(T)


            elif factorize=='TH-TW':
                f_cur_th = self.factorize(f1,dim=(2,3)) # (B,C*H)
                f_old_th = self.factorize(f2,dim=(2,3))
                f_cur_tw = self.factorize(f1,dim=(2,4)) # (B,C*W)
                f_old_tw = self.factorize(f2,dim=(2,4))
                f_cur = torch.cat([f_cur_th,f_cur_th],dim=-1)
                f_old = torch.cat([f_old_tw,f_old_tw],dim=-1)
                f_cur = F.normalize(f_cur, dim=1, p=2)
                f_old = F.normalize(f_old, dim=1, p=2)
                loss_i = torch.mean(torch.frobenius_norm(f_cur-f_old.clone().detach(),dim=-1))

            elif factorize=='T-H-W':
                f_cur_t = self.factorize(f1,dim=2) # (B,C*H*W)
                f_old_t = self.factorize(f2,dim=2)
                f_cur_h = self.factorize(f1,dim=3) # (B,C*T*W)
                f_old_h = self.factorize(f2,dim=3)
                f_cur_w = self.factorize(f1,dim=4) # (B,C*T*H)
                f_old_w = self.factorize(f2,dim=4)

                f_cur = torch.cat([f_cur_t,f_cur_h,f_cur_w],dim=-1)
                f_old = torch.cat([f_old_t,f_old_h,f_old_w],dim=-1)
                f_cur = F.normalize(f_cur, dim=1, p=2)
                f_old = F.normalize(f_old, dim=1, p=2)
                loss_i = torch.mean(torch.frobenius_norm(f_cur-f_old.clone().detach(),dim=-1))

            elif factorize=='all':
                f1 = f1.view(B,-1) # B, C*T*H*W
                f2 = f2.view(B,-1)
                f1 = F.normalize(f1, dim=1, p=2)
                f2 = F.normalize(f2, dim=1, p=2)
                loss_i = torch.mean(torch.frobenius_norm(f1-f2.clone().detach(),dim=-1))

            elif factorize=='spatial_pixel':
                f1 = f1.reshape((B,C*T,-1))
                f2 = f2.reshape((B,C*T,-1))
                f1 = F.normalize(f1, dim=2, p=2)
                f2 = F.normalize(f2, dim=2, p=2)
                if factor is not None: # Ours
                    factor_i = factor[i].permute(1,0)
                    factor_i = factor_i.reshape([1,-1])
                    loss_i = torch.mean(factor_i * torch.frobenius_norm(f1-f2.clone().detach(),dim=-1))
                else:
                    loss_i = torch.mean(torch.frobenius_norm(f1-f2.clone().detach(),dim=-1))
                loss_i = loss_i/sqrt(T)

            elif factorize=='pixel':
                #print(f1.size())
                #print(B,C,T,H,W)
                f1 = f1.reshape((B,C,-1)) # (B,C,T*H*W)
                f2 = f2.reshape((B,C,-1)) # (B,C,T*H*W)
                f1 = F.normalize(f1,dim=2,p=2)
                f2 = F.normalize(f2,dim=2,p=2)
                if factor is not None: 
                    factor_i = factor[i].reshape([1,-1]) # (1,C)
                    loss_i = torch.mean(factor_i * torch.frobenius_norm(f1-f2.clone().detach(), dim=-1))
                else:
                    loss_i = torch.mean(torch.frobenius_norm(f1-f2.clone().detach(),dim=-1))

            loss_dist = loss_dist + loss_i

        loss_dist = loss_dist/num_layers

        return loss_dist

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
        device = x.device
        num_seen_classes = len(self.seen_classes)
        offset_1, offset_2 = self._compute_offsets(self.cur_task_id)
        target = y
        assert y.shape[1] == offset_2 == num_seen_classes
        output, latent_feat, int_features = self.forward_net(x)
     
        assert output.shape[1] == num_seen_classes
        loss_1 = self.loss_classification(output, target)  # Lce Loss
     #   import pdb
    #    pdb.set_trace()
        if self.cur_task_id > 0:
            self.old_net.eval()
            with torch.no_grad():
                _, old_latent_feat, old_int_features = self.old_net(x)
                old_latent_feat = old_latent_feat.detach()

            loss_2 = self.lf_dist(latent_feat.clone(), old_latent_feat) * self.lambda_cur   #----------------
            loss_3 = self.feat_dist(int_features, old_int_features)
            loss_temp = 0.5 * loss_1  +  0.5 * loss_2 + 1.0 * loss_3  
        #    print(f'loss_temp:{loss_temp}, loss_1:{loss_1}, loss_2:{loss_2}, loss_3:{loss_3}')
            if loss_1 > 300:
                import pdb
                pdb.set_trace()
            
        else:
            loss_2 = torch.zeros_like(loss_1)
            loss_3 = torch.zeros_like(loss_1)
            loss_temp = loss_1 + loss_2 + loss_3

        if train:
            loss = loss_temp
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        else:
            loss = loss_1

        predictions = output.ge(0.0)

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

        output, _, _ = self.forward_net(x)
        assert output.shape[1] == num_seen_classes
        predictions = output.ge(0.0)
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
                        output, out_latent, _ = model.forward_net(images)
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
