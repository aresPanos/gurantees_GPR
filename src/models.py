# Copyright 2021 Aristeidis Panos

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple, TypeVar
import importlib.util
spec = importlib.util.find_spec('silence_tensorflow')
if spec is not None:
    import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K_bd
from itertools import product
import os 
from gpflow.base import Module, Parameter
from gpflow.config import default_float, set_default_float
from gpflow.utilities import ops, positive
from scipy import special

class MGP_model(Module):

    def __init__(self,
                 data: Tuple[tf.Tensor, tf.Tensor],
                 rank: int = 20,
                 dnn_out: int = 2,
                 eps_sq: float = 1,
                 sigma_n_sq: float = 1,
                 sigma_f_sq: float = 1):

        if data[1].dtype == np.float64:
            K_bd.set_floatx('float64')
        else:
            set_default_float(np.float32)

        self.num_data = tf.cast(data[1].shape[0], default_float())
        self.data = (tf.cast(tf.squeeze(data[0]), default_float()), tf.cast(data[1], default_float()))

        self.dim = dnn_out
        self.sqrt_2 = tf.cast(np.sqrt(2), default_float())
        self.minus_ln_2 = tf.cast(-1*np.log(2), default_float())

        num_eig_fun = 7
        all_tupples = np.asarray(list(product(range(1, num_eig_fun+1), repeat=self.dim)))
        args_tupples = np.argsort(all_tupples.sum(1))
        all_tupples = all_tupples[args_tupples]
        tupples = all_tupples[:rank]

        sum_tupples_D = tupples.sum(1) - self.dim
        self.sum_tupples_D = tf.cast(sum_tupples_D, default_float())

        self.tupples_1 = tf.cast(tupples - 1, tf.int32)
        self.max_rank = tupples.max()
        self.dim_half = 0.5 * self.dim

        self.this_range_1_2 = tf.range(self.max_rank, dtype=default_float())
        self.tf_range_D = tf.range(self.dim)
        self.gamma_term = np.exp(-0.5 * (special.gammaln(tupples).sum(1) + sum_tupples_D * np.log(2)))

        self.eye_k = tf.eye(rank, dtype=default_float())
        self.yTy = tf.reduce_sum(tf.math.square(self.data[1]))
        self.coeff_n_tf = tf.constant(np.load(os.path.dirname(os.path.realpath(__file__)) + '/hermite_coeff.npy')[:self.max_rank, :self.max_rank], dtype=default_float())

        self.rotation = Parameter(np.eye(self.dim), dtype=default_float())
        self.eps_sq = Parameter(eps_sq, transform=positive(), dtype=default_float(), trainable=True)
        self.sigma_f_sq = 1. #Parameter(sigma_f_sq, transform=positive(), dtype=default_float(), trainable=True)
        self.sigma_n_sq = Parameter(sigma_n_sq, transform=positive(), dtype=default_float(), trainable=True)

        model = models.Sequential()
        model.add(layers.Dense(dnn_out, activation='linear', input_dim=data[0].shape[1]))
        self.neural_net = model


    def neg_log_marginal_likelihood(self, data_in: tf.Tensor=None) -> tf.Tensor:
        if data_in is None:
            Xb = self.data[0]
            yb = self.data[1]
            yTy_b = self.yTy
        else:
            Xb = data_in[0]
            yb = data_in[1]
            yTy_b = tf.reduce_sum(tf.math.square(data_in[1]))

        data_rotated = self.neural_net(Xb) # [N, dnn_out]
        data_rotated = (data_rotated - tf.math.reduce_mean(data_rotated, 0)) / tf.math.reduce_std(data_rotated, 0)

        inv_sigma_sq = 1 / self.sigma_n_sq
        Lambda_Herm, V_herm = self.eigen_fun(data_rotated) # [rank, None], [N, rank]

        V_lambda_sqrt = V_herm * tf.math.sqrt(Lambda_Herm) # [N, rank]
        V_lambda_sqrt_y = tf.linalg.matvec(V_lambda_sqrt, yb, transpose_a=True) # [rank, None]

        low_rank_term = self.eye_k + inv_sigma_sq * tf.linalg.matmul(V_lambda_sqrt, V_lambda_sqrt, transpose_a=True) # [rank, rank]
        low_rank_L = tf.linalg.cholesky(low_rank_term)
        L_inv_V_y = tf.linalg.triangular_solve(low_rank_L, V_lambda_sqrt_y[:, None], lower=True) # [rank, 1]
        data_fit = inv_sigma_sq * (self.yTy - inv_sigma_sq * tf.reduce_sum(tf.math.square(L_inv_V_y)))
        return data_fit + self.num_data * tf.math.log(self.sigma_n_sq) + 2. * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(low_rank_L)))


    def eigen_fun(self, data_x) -> Tuple[tf.Tensor, tf.Tensor]:
        beta = tf.pow(1 + 8 * self.eps_sq, .25)
        delta_sq = 0.25 * (tf.square(beta) - 1.)
        log_a_d_ep_sq = tf.math.log(0.5 + delta_sq + self.eps_sq)

        log_term = self.dim_half * (self.minus_ln_2 - log_a_d_ep_sq) + self.sum_tupples_D * (tf.math.log(self.eps_sq) - log_a_d_ep_sq) # [rank, None]
        gamma = tf.pow(beta, self.dim_half) * self.gamma_term # [rank, None]

        tmp_exp = tf.exp(-delta_sq * ( tf.reduce_sum(tf.square(data_x), axis=1))) # [N, None]
        x_data_upgr = beta * data_x / self.sqrt_2 # [N, D]

        vander_tf = tf.math.pow(tf.expand_dims(x_data_upgr, axis=-1), self.this_range_1_2[None, :]) # [N, D, max_rank]
        all_hermite_pols = tf.linalg.matmul(vander_tf, self.coeff_n_tf, transpose_b=True) # [N, D, max_rank]

        eigen_fun_tmp = tf.map_fn(lambda x: tf.gather(all_hermite_pols[:, x], self.tupples_1[:, x], axis=-1), self.tf_range_D, dtype=default_float()) # [D, N, rank]
        tf_lambda_n = tf.exp(log_term) # [rank, None]
        tf_phi_n = tf.reduce_prod(eigen_fun_tmp, 0) # [N, rank]
        tf_phi_n *= gamma # [N, rank]
        tf_phi_n *= tmp_exp[:, None] # [N, rank]

        return self.sigma_f_sq * tf_lambda_n, tf_phi_n


    def predict_y(self, x_test: tf.Tensor, full_cov: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        inv_sigma_sq = 1/self.sigma_n_sq

        data_rotated_tr = self.neural_net(self.data[0])
        mean_x_nn_tr, std_x_nn_tr = tf.math.reduce_mean(data_rotated_tr, 0), tf.math.reduce_std(data_rotated_tr, 0)
        data_rotated_tr = (data_rotated_tr - mean_x_nn_tr) / std_x_nn_tr

        data_rotated_tst = self.neural_net(x_test)
        data_rotated_tst = (data_rotated_tst - mean_x_nn_tr) / std_x_nn_tr

        Lambda_Herm, V_herm = self.eigen_fun(data_rotated_tr) # [N, k]
        V_herm_test = self.eigen_fun(data_rotated_tst)[1] # [Ntest, k]
        sqrt_Lambda_Herm = tf.math.sqrt(Lambda_Herm) # [k, None]

        V_lambda_sqrt = V_herm*sqrt_Lambda_Herm # [N, k]
        V_lambda_sqrt_y = tf.linalg.matvec(V_lambda_sqrt, self.data[1], transpose_a=True) # [k, None]

        V_test_lambda_sqrt = V_herm_test*sqrt_Lambda_Herm # [Ntest, k]
        K_Xtest_X_y =  tf.linalg.matvec(V_test_lambda_sqrt, V_lambda_sqrt_y) # [Ntest, None]
        VT_V = tf.linalg.matmul(V_lambda_sqrt, V_lambda_sqrt, transpose_a=True) # [k, k]

        low_rank_term = self.eye_k + inv_sigma_sq*VT_V # [k, k]
        low_rank_L = tf.linalg.cholesky(low_rank_term)
        L_inv_V_y = tf.linalg.triangular_solve(low_rank_L, V_lambda_sqrt_y[:, None], lower=True) # [k, 1]

        V_K_Xtest = tf.linalg.matmul(VT_V, V_test_lambda_sqrt, transpose_b=True) # [k, N_test]
        tmp_inv = tf.linalg.triangular_solve(low_rank_L, V_K_Xtest, lower=True) # [k, N_test]
        mean_f = inv_sigma_sq*(K_Xtest_X_y - inv_sigma_sq*tf.linalg.matvec(tmp_inv, tf.squeeze(L_inv_V_y), transpose_a=True))

        if full_cov:
            tmp_matmul = tf.linalg.matmul(V_test_lambda_sqrt, V_K_Xtest) # [Ntest, Ntest]
            K_Xtest_herm = tf.linalg.matmul(V_test_lambda_sqrt, V_test_lambda_sqrt, transpose_b=True) # [Ntest, Ntest]
            var_f = K_Xtest_herm - inv_sigma_sq*(tmp_matmul - inv_sigma_sq*tf.linalg.matmul(tmp_inv, tmp_inv, transpose_a=True))
            diag = tf.linalg.diag_part(var_f) + self.sigma_n_sq
            var_f = tf.linalg.set_diag(var_f, diag)
        else:
            var_f = self.sigma_f_sq + self.sigma_n_sq - inv_sigma_sq*(tf.einsum('kn,nk->n', V_K_Xtest, V_test_lambda_sqrt) - inv_sigma_sq*tf.reduce_sum(tf.math.square(tmp_inv), 0))

        return mean_f, var_f
       
           
class FGP_model(Module):

    def __init__(self, 
                 data: Tuple[tf.Tensor, tf.Tensor],  
                 m: int = 100, 
                 lengthscales = None,
                 sigma_n_sq: np.float = 1,
                 sigma_f_sq: np.float = 1,
                 randn = None):
                                
        self.num_data = tf.cast(data[1].size, default_float())
        self.data = (tf.cast(data[0], default_float()), tf.cast(data[1], default_float()))
        self.const = tf.cast(0.5*data[1].size*np.log(2*np.pi), default_float())
                       
        self.eye_2m = tf.eye(2*m, dtype=default_float())
        self.yTy = tf.reduce_sum(tf.math.square(self.data[1])) 
        self.m_float = tf.cast(m, default_float())
        self.randn = tf.random.normal(shape=[m, data[0].shape[1]], dtype=default_float()) if randn is None else tf.cast(randn[:, None], default_float())
        
        lengthscales0 = np.ones(data[0].shape[1]) if lengthscales is None else lengthscales
        self.lengthscales = Parameter(lengthscales0, transform=positive(), dtype=default_float())
        self.sigma_f_sq = Parameter(sigma_f_sq, transform=positive(), dtype=default_float())
        self.sigma_n_sq = Parameter(sigma_n_sq, transform=positive(), dtype=default_float())
       
       
    def neg_log_marginal_likelihood(self) -> tf.Tensor:   
        '''
        It computes the negative log marginal likelihood up to a constant
        '''    
        inv_sigma_sq = 1/self.sigma_n_sq 
        V_lambda_sqrt = self.fourier_features(self.data[0]) # [batch, 2m]
        V_lambda_sqrt_y = tf.linalg.matvec(V_lambda_sqrt, self.data[1], transpose_a=True) # [k, None]
        
        low_rank_term = self.eye_2m + inv_sigma_sq*tf.linalg.matmul(V_lambda_sqrt, V_lambda_sqrt, transpose_a=True) # [k, k]
        low_rank_L = tf.linalg.cholesky(low_rank_term)
        L_inv_V_y = tf.linalg.triangular_solve(low_rank_L, V_lambda_sqrt_y[:, None], lower=True) # [k, 1]
        data_fit = inv_sigma_sq*(self.yTy - inv_sigma_sq*tf.reduce_sum(tf.math.square(L_inv_V_y)))
        return 0.5*(data_fit + self.num_data*tf.math.log(self.sigma_n_sq)) + tf.reduce_sum(tf.math.log(tf.linalg.diag_part(low_rank_L))) + self.const
                        
        
    def fourier_features(self, data_x) -> tf.Tensor:  
        freq = self.randn / self.lengthscales
        xall_freq = tf.linalg.matmul(data_x, freq, transpose_b=True) # [batch, m]
        cos_freq = tf.math.cos(xall_freq)
        sin_freq = tf.math.sin(xall_freq) 
        full_z = tf.concat([cos_freq, sin_freq], 1) # [batch, 2m]
        
        return tf.math.sqrt(self.sigma_f_sq/self.m_float)*full_z
    
       
    def predict_y(self, x_test: tf.Tensor, full_cov: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:  
        '''
        It computes the mean and variance of the held-out data x_test.
        
        :x_test: tf.Tensor
            Input locations of the held-out data with shape=[Ntest, D]
            where Ntest is the number of rows and D is the input dimension of each point.
        :full_cov: bool
            If True, compute and return the full Ntest x Ntest test covariance matrix. Otherwise return only the diagonal of this matrix.
        '''  
        
        inv_sigma_sq = 1/self.sigma_n_sq 
              
        V_lambda_sqrt = self.fourier_features(self.data[0]) # [N, 2m]
        V_lambda_sqrt_y = tf.linalg.matvec(V_lambda_sqrt, self.data[1], transpose_a=True) # [2m, None]

        V_test_lambda_sqrt = self.fourier_features(x_test) # [Ntest, 2m]
        K_Xtest_X_y =  tf.linalg.matvec(V_test_lambda_sqrt, V_lambda_sqrt_y) # [Ntest, None]
        VT_V = tf.linalg.matmul(V_lambda_sqrt, V_lambda_sqrt, transpose_a=True) # [2m, 2m]
        
        low_rank_term = self.eye_2m + inv_sigma_sq*VT_V # [2m, 2m]
        low_rank_L = tf.linalg.cholesky(low_rank_term)
        L_inv_V_y = tf.linalg.triangular_solve(low_rank_L, V_lambda_sqrt_y[:, None], lower=True) # [2m, 1]
        
        V_K_Xtest = tf.linalg.matmul(VT_V, V_test_lambda_sqrt, transpose_b=True) # [2m, N_test]
        tmp_inv = tf.linalg.triangular_solve(low_rank_L, V_K_Xtest, lower=True) # [2m, N_test]
        mean_f = inv_sigma_sq*(K_Xtest_X_y - inv_sigma_sq*tf.linalg.matvec(tmp_inv, tf.squeeze(L_inv_V_y), transpose_a=True))
            
        if full_cov:     
            tmp_matmul = tf.linalg.matmul(V_test_lambda_sqrt, V_K_Xtest) # [Ntest, Ntest]
            K_Xtest_herm = tf.linalg.matmul(V_test_lambda_sqrt, V_test_lambda_sqrt, transpose_b=True) # [Ntest, Ntest]
            var_f = K_Xtest_herm - inv_sigma_sq*(tmp_matmul - inv_sigma_sq*tf.linalg.matmul(tmp_inv, tmp_inv, transpose_a=True))  
            diag = tf.linalg.diag_part(var_f) + self.sigma_n_sq
            var_f = tf.linalg.set_diag(var_f, diag)            
        else:
            var_f = self.sigma_f_sq + self.sigma_n_sq - inv_sigma_sq*(tf.einsum('kn,nk->n', V_K_Xtest, V_test_lambda_sqrt) - inv_sigma_sq*tf.reduce_sum(tf.math.square(tmp_inv), 0))
            
        return mean_f, var_f
        