import torch
from misc_utils import safe_log as log
from misc_utils import safe_exp as exp
# from misc_utils import safe_sqrt as sqrt

import numpy as np

# class RBF():
#     def __init__(self, variance=1., lengthscale=0.5):
#         variance = np.log(variance)
#         lengthscale = np.log(lengthscale)
#         # self.variance = torch.tensor(variance, requires_grad=True)
#         # self.lengthscale = torch.tensor(lengthscale, requires_grad=True)
#         self.variance = torch.tensor(variance)
#         self.lengthscale = torch.tensor(lengthscale)
        
#     def K(self, X1, X2):
#         return exp(self.variance) * exp(-0.5 * (self._euclidean_distance(X1, X2) / exp(self.lengthscale))**2)
        
#     def _euclidean_distance(self, X1, X2):
#         X1sq = torch.sum(X1**2, dim=1)
#         X2sq = torch.sum(X2**2, dim=1)
#         r = -2. * (X1 @ X2.t()) + (X1sq[:, None] + X2sq[None, :])
#         r = sqrt(r)
#         return r
    
    
class RBF():
    def __init__(self, variance=1., lengthscale=0.5):
        variance = np.log(variance)
        lengthscale = np.log(lengthscale)
        self.variance = torch.tensor(variance, requires_grad=True)
        self.lengthscale = torch.tensor(lengthscale, requires_grad=True)
        
    def K(self, X1, X2):
        distance = torch.cdist(X1, X2)  
        scaled_distance = distance / exp(self.lengthscale)
        return exp(self.variance) * exp(-0.5 * scaled_distance**2)
