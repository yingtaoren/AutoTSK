import itertools
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional
from cuml.cluster import KMeans


from utils.util import check_tensor
import importlib.util

package_name = "pandas"

spec = importlib.util.find_spec(package_name)
if spec is None:
    print(package_name + " is not installed")



def antecedent_init_center_class_kmeans(X, y, class_num=2):

    km = KMeans(n_clusters=class_num,random_state = 2025)
    result = km.fit_predict(X)

    n_features = X.shape[1]
    cluster_variances = np.zeros((class_num, n_features))

    for i in range(class_num):
        cluster_samples = X[result == i]  
        variances = np.mean((cluster_samples - km.cluster_centers_[i]) ** 2, axis=0)
        cluster_variances[i] = np.sqrt(variances) 

    return km.cluster_centers_.T, cluster_variances.T

def antecedent_init_center(
    X, y=None, n_rule=2, method="kmean", engine="sklearn", n_init=20
):

    if method == "kmean":

        km = KMeans(n_clusters=n_rule,random_state=2025)
        result = km.fit_predict(X)

        n_features = X.shape[1]
        cluster_variances = np.zeros((n_rule, n_features))

        for i in range(n_rule):
            cluster_samples = X[result == i] 
            variances = np.mean((cluster_samples - km.cluster_centers_[i]) ** 2, axis=0)
            cluster_variances[i] = np.sqrt(variances) 

        return km.cluster_centers_.T, cluster_variances.T


class Antecedent(nn.Module):
    def forward(self, **kwargs):
        raise NotImplementedError

    def init(self, **kwargs):
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError

class AntecedentSONFIN(Antecedent):

    def __init__(
        self,
        in_dim,
        n_rule,
        high_dim=False,
        init_center=None,
        init_sigma=1.0,
        eps=1e-16,
    ):
        super(AntecedentSONFIN, self).__init__()
        self.in_dim = in_dim
        self.n_rule = n_rule
        self.high_dim = high_dim

        self.init_center = (
            check_tensor(init_center, torch.float32)
            if init_center is not None
            else None
        )
        self.init_sigma = (
            check_tensor(init_sigma, torch.float32)
            if init_sigma is not None
            else torch.ones(size=(in_dim, n_rule))
        )
        # self.zr_op = torch.mean if high_dim else torch.sum
        self.eps = eps
        self.using_alpha = False
        self.using_alpha_gate = False
        self.alpha_gate = None
        self.alpha = None
        # input, target = input.to(self.device), target.to(self.device)

        self.__build_model__()

    def __build_model__(self):
        self.center = nn.Parameter(torch.zeros(size=(self.in_dim, self.n_rule)))
        self.sigma = nn.Parameter(torch.zeros(size=(self.in_dim, self.n_rule)))

        self.alpha = nn.Parameter(
            torch.full(size=(self.in_dim, self.n_rule), fill_value=1.0)
        )

        self.reset_parameters()  # init_centers to be set

    def set_using_alpha(self):
        self.using_alpha = True
        self.using_alpha_gate = False
        
    def set_using_alpha_gate(self):
        self.using_alpha_gate = True
        self.using_alpha = False

    def init(self, center, sigma, alpha):

        center = check_tensor(center, torch.float32)
        self.init_center = center
        self.init_sigma = sigma

        self.reset_parameters()

    def reset_parameters(self):

        if self.init_sigma is not None:
            if isinstance(self.init_sigma, torch.Tensor) and self.init_sigma.dim() > 0:
                self.sigma.data[...] = torch.FloatTensor(self.init_sigma)
            else:
                self.sigma.data.fill_(self.init_sigma)
        else:
            self.sigma.data.fill_(1.0)

        if self.init_center is not None:
            self.center.data[...] = torch.FloatTensor(self.init_center)
        else:
            init.normal_(self.center, 0, 1)



    def forward(self, X, original_frs=False):

        if X.dim() < 2:
            X = X.unsqueeze(0) if X.dim() == 1 else X.unsqueeze(0).unsqueeze(1)
        temp = None
        if self.using_alpha:
            exponent = (
                -((X[:, :, None] - self.center[None, :, :]) ** 2)
                / (2.0 * (self.sigma[None, :, :] ** 2 + self.eps))
            ) * (1 / (self.alpha ** 2 +self.eps))

            temp = exponent.sum(dim=1, keepdim=False) / self.in_dim
        elif self.using_alpha_gate:
            exponent = (
                -((X[:, :, None] - self.center[None, :, :]) ** 2)
                / (2.0 * (self.sigma[None, :, :] ** 2 + self.eps))
            ) * self.alpha_gate


            remain_mf_number = self.alpha_gate.count_nonzero(dim=0) 
            sum_of_exponent = exponent.sum(dim=1, keepdim=False)
            correct0_sem_of_exponent = torch.where(sum_of_exponent == 0, torch.tensor(-99.0, device=sum_of_exponent.device), sum_of_exponent)

            temp = correct0_sem_of_exponent * (1 / (remain_mf_number + self.eps))

        frs = functional.softmax(temp, dim=1)
        if original_frs:
            return frs, temp
        return frs
