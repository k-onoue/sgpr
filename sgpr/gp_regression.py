import torch
from torch import pi 
torch.set_default_dtype(torch.float64)

from .gp_utils import RBF
from .linalg_utils import det, inv, inv_diag
from .misc_utils import safe_log as log
from .misc_utils import safe_exp as exp

import numpy as np
import pandas as pd


class _base_gp():
    def __init__(self, X_train, y_train, kernel=None, noise=1.):
        self.X_train = X_train.requires_grad_(True)
        self.y_train = y_train.requires_grad_(True)
        noise = np.log(noise)
        self.noise = torch.tensor(noise, requires_grad=True)

        if kernel is None:
            kernel = RBF()
        self.kernel = kernel

        self.params = [self.noise, self.kernel.variance, self.kernel.lengthscale]

        self.params_opt_results = []

        #####################################3
        # self.pseudo_inputs = None

    def _calc_misc_for_prediction(self, X_pred):
        X_pred = X_pred
        K_us = None
        K_su = K_us.T
        K_uu = None
        i_K_uu = inv(K_uu) 
        Sigma = None
        K_uf = None
        i_Lambda = None
        P_ss = None
        return K_us, K_su, i_K_uu, Sigma, K_uf, i_Lambda, P_ss
    
    def predict(self, X_pred):
        K_us, K_su, i_K_uu, Sigma, K_uf, i_Lambda, P_ss = self._calc_misc_for_prediction(X_pred)
        y = self.y_train
        mean = K_su @ Sigma @ K_uf @ i_Lambda @ y
        cov = (P_ss - K_su @ i_K_uu @ K_us) + K_su @ Sigma @ K_us
        return mean, cov

    def _calc_misc_for_optimization(self):
        K_uu = None
        i_K_uu = None
        K_uf = None
        K_fu = K_uf.T
        Lambda = None
        i_Lambda = None
        Sigma = None
        return K_uu, i_K_uu, K_uf, K_fu, Lambda, i_Lambda, Sigma

    def _log_marginal_likelihood(self):
        K_uu, i_K_uu, K_uf, K_fu, Lambda, i_Lambda, Sigma = self._calc_misc_for_optimization()
        y = self.y_train
        n = y.shape[0]
        I = torch.eye(n)
        term1 = - (1/2) * log(det(K_uu + K_uf @ i_Lambda @ K_fu) * det(i_K_uu) * det(Lambda)) 
        term2 = - (1/2) * y.T @ i_Lambda @ (I + K_fu @ Sigma @ K_uf @ i_Lambda) @ y 
        term3 = - (n/2) * log(torch.tensor(2 * pi))
        return term1 + term2 + term3
    
    def _store_params(self):
        # tmp_params_1 = [np.exp(param.item()) for param in self.params[:3]] 
        tmp_params_1 = [param.item() for param in self.params[:3]] 

        if len(self.params) > 3:
            tmp_params_2 = [param.item() for param in list(self.params[3])]
        else:
            tmp_params_2 = []

        self.params_opt_results.append(tmp_params_1 + tmp_params_2)

        # tmp_params = [param.item() for param in self.params]
        # self.params_opt_results.append(tmp_params)

    def make_params_df(self):
        col_name = ["noise", "variance", "lengthscale"]

        if len(self.params) > 3:
            tmp_name = [f'u_{i+1}' for i in range(len(self.params[3]))]
        else:
            tmp_name = []

        col_name = col_name + tmp_name

        tmp_arr = np.array(self.params_opt_results)
        tmp_arr[:, :3] = np.exp(tmp_arr[:, :3])
        return pd.DataFrame(tmp_arr, columns=col_name)
        
    
    def optimize(self, iteration=10, learning_rate=0.1):
        optimizer = torch.optim.Adam(self.params, lr=learning_rate) 

        self._store_params()


        for i in range(iteration):
            print(f'opt_iter: {i+1}/{iteration}')

            optimizer.zero_grad()

            try:
                loss = self._log_marginal_likelihood()
                # loss = - self._log_marginal_likelihood()
                loss.backward()
                optimizer.step()

                self._store_params()

            except Exception as e:
                print(f"An error occurred during optimization: {e}")
                print("Using previous parameters and ending optimization.")
                
                tmp1 = self.params_opt_results[i-1][:3]
                self.noise = torch.tensor(tmp1[0], requires_grad=True)
                self.kernel.variance = torch.tensor(tmp1[1], requires_grad=True)
                self.kernel.lengthscale = torch.tensor(tmp1[2], requires_grad=True)
                
                for j, new_param in enumerate([self.noise, self.kernel.variance, self.kernel.lengthscale]):
                    self.params[j] = new_param
                
                if len(self.params) > 3:
                    tmp2 = self.params_opt_results[i-1][3:]
                    self.pseudo_inputs = torch.Tensor(tmp2).reshape(-1, 1).requires_grad_(True) 
                    self.params[3] = self.pseudo_inputs  
                
                break
            
            # self._store_params()
    
    
class GP(_base_gp):
    def __init__(self, X_train, y_train, kernel=None, noise=1.):
        super().__init__(X_train, y_train, kernel, noise)
        self.name = "GP regression"
    
    def _calc_misc_for_prediction(self, X_pred):
        f = self.X_train
        u = self.X_train
        s = X_pred
        noise = exp(self.noise)

        K_us = self.kernel.K(u, s)
        K_su = K_us.T
        K_uu = self.kernel.K(u, u)
        i_K_uu = inv(K_uu)
        K_uf = self.kernel.K(u, f)
        K_fu = K_uf.T
        K_ss = self.kernel.K(s, s)

        Lambda = noise * torch.eye(f.shape[0])
        i_Lambda = inv_diag(Lambda)
        i_Sigma = K_uu + K_uf @ i_Lambda @ K_fu
        Sigma = inv(i_Sigma)

        P_ss = K_ss

        return K_us, K_su, i_K_uu, Sigma, K_uf, i_Lambda, P_ss

    def _calc_misc_for_optimization(self):
        f = self.X_train
        u = self.X_train
        noise = exp(self.noise)
        
        K_uu = self.kernel.K(u, u)
        i_K_uu = inv(K_uu)
        K_uf = self.kernel.K(u, f)
        K_fu = K_uf.T

        Lambda = noise * torch.eye(f.shape[0])
        i_Lambda = inv_diag(Lambda)
        i_Sigma = K_uu + K_uf @ i_Lambda @ K_fu
        Sigma = inv(i_Sigma)

        return K_uu, i_K_uu, K_uf, K_fu, Lambda, i_Lambda, Sigma    


class _base_sparse_gp(_base_gp):
    def __init__(self, X_train, y_train, pseudo_inputs=None, p_inputs_num=None, p_optimized=False, kernel=None, noise=torch.tensor(1.)):
        super().__init__(X_train, y_train, kernel, noise)
        if pseudo_inputs is None:
            if not p_inputs_num:
                size = 10
            else:
                size = p_inputs_num
            pseudo_inputs = torch.linspace(X_train.min().item(), X_train.max().item(), size).reshape(-1, 1)
            self.pseudo_inputs = pseudo_inputs.requires_grad_(True)
        else:
            self.pseudo_inputs = pseudo_inputs.requires_grad_(True)

        if p_optimized is True:
            self.params = [self.noise, self.kernel.variance, self.kernel.lengthscale, self.pseudo_inputs]
    

class SoR(_base_sparse_gp):
    def __init__(self, X_train, y_train, pseudo_inputs=None, p_inputs_num=None, p_optimized=False, kernel=None, noise=1.):
        super().__init__(X_train, y_train, pseudo_inputs, p_inputs_num, p_optimized, kernel, noise)

        self.name = "SoR"

    def _calc_misc_for_prediction(self, X_pred):
        f = self.X_train
        u = self.pseudo_inputs
        s = X_pred
        noise = exp(self.noise)

        K_us = self.kernel.K(u, s)
        K_su = K_us.T
        K_uu = self.kernel.K(u, u)
        i_K_uu = inv(K_uu)
        K_uf = self.kernel.K(u, f)
        K_fu = K_uf.T
        # K_ss = self.kernel.K(s, s)

        Lambda = noise * torch.eye(f.shape[0])
        i_Lambda = inv_diag(Lambda)
        i_Sigma = K_uu + K_uf @ i_Lambda @ K_fu
        Sigma = inv(i_Sigma)

        P_ss = K_su @ i_K_uu @ K_us

        return K_us, K_su, i_K_uu, Sigma, K_uf, i_Lambda, P_ss
    
    def _calc_misc_for_optimization(self):
        f = self.X_train
        u = self.pseudo_inputs
        noise = exp(self.noise)

        K_uu = self.kernel.K(u, u)
        i_K_uu = inv(K_uu)
        K_uf = self.kernel.K(u, f)
        K_fu = K_uf.T

        Lambda = noise * torch.eye(f.shape[0])
        i_Lambda = inv_diag(Lambda)
        i_Sigma = K_uu + K_uf @ i_Lambda @ K_fu
        Sigma = inv(i_Sigma) 

        return K_uu, i_K_uu, K_uf, K_fu, Lambda, i_Lambda, Sigma
    

class DTC(_base_sparse_gp):
    def __init__(self, X_train, y_train, pseudo_inputs=None, p_inputs_num=None, p_optimized=False, kernel=None, noise=1.):
        super().__init__(X_train, y_train, pseudo_inputs, p_inputs_num, p_optimized, kernel, noise)

        self.name = "DTC"

    def _calc_misc_for_prediction(self, X_pred):
        f = self.X_train
        u = self.pseudo_inputs
        s = X_pred
        noise = exp(self.noise)

        K_us = self.kernel.K(u, s)
        K_su = K_us.T
        K_uu = self.kernel.K(u, u)
        i_K_uu = inv(K_uu)
        K_uf = self.kernel.K(u, f)
        K_fu = K_uf.T
        K_ss = self.kernel.K(s, s)

        Lambda = noise * torch.eye(f.shape[0])
        i_Lambda = inv_diag(Lambda)
        i_Sigma = K_uu + K_uf @ i_Lambda @ K_fu
        Sigma = inv(i_Sigma)

        P_ss = K_ss

        return K_us, K_su, i_K_uu, Sigma, K_uf, i_Lambda, P_ss
    
    def _calc_misc_for_optimization(self):
        f = self.X_train
        u = self.pseudo_inputs
        noise = exp(self.noise)
        K_uu = self.kernel.K(u, u)
        i_K_uu = inv(K_uu)
        K_uf = self.kernel.K(u, f)
        K_fu = K_uf.T

        Lambda = noise * torch.eye(f.shape[0])
        i_Lambda = inv_diag(Lambda)
        i_Sigma = K_uu + K_uf @ i_Lambda @ K_fu
        Sigma = inv(i_Sigma)

        return K_uu, i_K_uu, K_uf, K_fu, Lambda, i_Lambda, Sigma
    

class FITC(_base_sparse_gp):
    def __init__(self, X_train, y_train, pseudo_inputs=None, p_inputs_num=None, p_optimized=False, kernel=None, noise=1.):
        super().__init__(X_train, y_train, pseudo_inputs, p_inputs_num, p_optimized, kernel, noise)

        self.name = "FITC"

    def _calc_misc_for_prediction(self, X_pred):
        f = self.X_train
        u = self.pseudo_inputs
        s = X_pred
        noise = exp(self.noise)

        K_us = self.kernel.K(u, s)
        K_su = K_us.T
        K_uu = self.kernel.K(u, u)
        i_K_uu = inv(K_uu)
        K_uf = self.kernel.K(u, f)
        K_fu = K_uf.T
        K_ss = self.kernel.K(s, s)
        K_ff = self.kernel.K(f, f)
        
        Q_ff = K_fu @ i_K_uu @ K_uf

        Lambda = torch.diag(torch.diagonal(K_ff - Q_ff + noise * torch.eye(f.shape[0])))
        # Lambda = torch.diag(torch.diagonal(torch.eye(f.shape[0]) - Q_ff + noise * torch.eye(f.shape[0])))
        i_Lambda = inv_diag(Lambda)
        i_Sigma = K_uu + K_uf @ i_Lambda @ K_fu
        Sigma = inv(i_Sigma)

        P_ss = K_ss

        return K_us, K_su, i_K_uu, Sigma, K_uf, i_Lambda, P_ss
    
    def _calc_misc_for_optimization(self):
        f = self.X_train
        u = self.pseudo_inputs
        noise = exp(self.noise)

        K_uu = self.kernel.K(u, u)
        i_K_uu = inv(K_uu)
        K_uf = self.kernel.K(u, f)
        K_fu = K_uf.T

        Lambda = noise * torch.eye(f.shape[0])
        i_Lambda = inv_diag(Lambda)
        i_Sigma = K_uu + K_uf @ i_Lambda @ K_fu
        Sigma = inv(i_Sigma)

        return K_uu, i_K_uu, K_uf, K_fu, Lambda, i_Lambda, Sigma