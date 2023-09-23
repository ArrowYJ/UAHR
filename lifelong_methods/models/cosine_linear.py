'''
Adapted from
https://github.com/hshustc/CVPR19_Incremental_Learning/blob/master/cifar100-class-incremental/modified_linear.py

Reference:
[1] Saihui Hou, Xinyu Pan, Chen Change Loy, Zilei Wang, Dahua Lin
    Learning a Unified Classifier Incrementally via Rebalancing. CVPR 2019
'''

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import Module
from typing import Union

def stable_cosine_distance(a, b, squared=True):
    # From PODNet
    """Computes the pairwise distance matrix with numerical stability."""
    mat = torch.cat([a, b])

    pairwise_distances_squared = torch.add(
        mat.pow(2).sum(dim=1, keepdim=True).expand(mat.size(0), -1),
        torch.t(mat).pow(2).sum(dim=0, keepdim=True).expand(mat.size(0), -1)
    ) - 2 * (torch.mm(mat, torch.t(mat)))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.clamp(pairwise_distances_squared, min=0.0)

    # Get the mask where the zero distances are at.
    error_mask = torch.le(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(pairwise_distances_squared + error_mask.float() * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = torch.mul(pairwise_distances, (error_mask == False).float())

    # Explicitly set diagonals to zero.
    mask_offdiagonals = 1 - torch.eye(*pairwise_distances.size(), device=pairwise_distances.device)
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    return pairwise_distances[:a.shape[0], a.shape[0]:]

def _reduce_proxies(similarities, num_proxy):
    # shape (batch_size, n_classes * proxy_per_class)
    n_classes = similarities.shape[1] / num_proxy
    assert n_classes.is_integer(), (similarities.shape[1], num_proxy)
    n_classes = int(n_classes)
    bs = similarities.shape[0]

    simi_per_class = similarities.view(bs, n_classes, num_proxy)
    attentions = F.softmax(simi_per_class, dim=-1)  # shouldn't be -gamma?
    return (simi_per_class * attentions).sum(-1)

class CosineLinearProxy(Module):
    def __init__(self, in_features, out_features, num_proxy=1, sigma: Union[bool, float, int] = True):
        super(CosineLinearProxy, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_proxy = num_proxy
        self.weight = Parameter(torch.empty(self.num_proxy * out_features, in_features), requires_grad=True)
        if isinstance(sigma, bool):
            if sigma:
                self.sigma = Parameter(torch.empty(1), requires_grad=True)
                self.sigma.data.fill_(1)
            else:
                self.register_parameter('sigma', None)
        elif isinstance(sigma, int) or isinstance(sigma, float):
            self.register_buffer('sigma', torch.tensor(float(sigma)))
        else:
            raise ValueError("sigma should be a boolean or a float")
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input_: torch.Tensor):
    #    out = F.linear(F.normalize(input_, p=2,dim=1), F.normalize(self.weight, p=2, dim=1))
        import pdb
        try:
            features = self.sigma * F.normalize(input_,p=2,dim=1)
        except:
            print(input_.shape)
            pdb.set_trace()
        weights = self.sigma * F.normalize(self.weight,p=2,dim=1)
        out = F.linear(features, weights)
    #    out =  stable_cosine_distance(features, weights)
        out = _reduce_proxies(out, self.num_proxy)
        if self.sigma is not None:
            out = self.sigma * out
        return out


class SplitCosineLinearProxy(Module):
    #consists of two fc layers and concatenate their outputs
    def __init__(self, in_features, out_features1, out_features2, num_proxy = 1, sigma: Union[bool, float, int] = True):
        super(SplitCosineLinearProxy, self).__init__()
        self.num_proxy = num_proxy

        self.in_features = in_features
        self.out_features = out_features1 + out_features2

        self.fc1 = CosineLinearProxy(in_features, out_features1, self.num_proxy, True)
        self.fc2 = CosineLinearProxy(in_features, out_features2, self.num_proxy, True)
        if isinstance(sigma, bool):
            if sigma:
                self.sigma = Parameter(torch.empty(1), requires_grad=True)
                self.sigma.data.fill_(1)
            else:
                self.register_parameter('sigma', None)
        elif isinstance(sigma, int) or isinstance(sigma, float):
            self.register_buffer('sigma', torch.tensor(float(sigma)))
        else:
            raise ValueError("sigma should be a boolean or a float")

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out = torch.cat((out1, out2), dim=1) # concatenate along the channel
        if self.sigma is not None:
            out = self.sigma * out
        return out


class CosineLinear(Module):
    def __init__(self, in_features, out_features, sigma: Union[bool, float, int] = True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(out_features, in_features), requires_grad=True)
        if isinstance(sigma, bool):
            if sigma:
                self.sigma = Parameter(torch.empty(1), requires_grad=True)
                self.sigma.data.fill_(1)
            else:
                self.register_parameter('sigma', None)
        elif isinstance(sigma, int) or isinstance(sigma, float):
            self.register_buffer('sigma', torch.tensor(float(sigma)))
        else:
            raise ValueError("sigma should be a boolean or a float")
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input_: torch.Tensor):
        out = F.linear(F.normalize(input_, p=2,dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out


class SplitCosineLinear(Module):
    #consists of two fc layers and concatenate their outputs
    def __init__(self, in_features, out_features1, out_features2, sigma: Union[bool, float, int] = True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features1 + out_features2
        self.fc1 = CosineLinear(in_features, out_features1, False)
        self.fc2 = CosineLinear(in_features, out_features2, False)
        if isinstance(sigma, bool):
            if sigma:
                self.sigma = Parameter(torch.empty(1), requires_grad=True)
                self.sigma.data.fill_(1)
            else:
                self.register_parameter('sigma', None)
        elif isinstance(sigma, int) or isinstance(sigma, float):
            self.register_buffer('sigma', torch.tensor(float(sigma)))
        else:
            raise ValueError("sigma should be a boolean or a float")

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out = torch.cat((out1, out2), dim=1) # concatenate along the channel
        if self.sigma is not None:
            out = self.sigma * out
        return out
