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

import importlib.util
spec = importlib.util.find_spec('silence_tensorflow')
if spec is not None:
    import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import os 
import gpflow
from models import MGP_model, FGP_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

  
    
def load_dataset_train_test_split(dataset_name, test_size = 0.1, split_id: int = 1, print_dataset: bool=False):
    if dataset_name not in ['elevators', 'protein', 'sarcos', '3droad']:
        raise NameError('The dataset with name ' + dataset_name + ' is not valid. Available dataset names: elevators, protein, sarcos, 3droad.')

    dir_parent = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
    data_all = np.load(dir_parent + '/data/' + dataset_name + '_all.npy', allow_pickle=True)
    x_all, y_all = data_all[()]['x_all'], data_all[()]['y_all']
    
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=test_size, random_state=split_id)
    if print_dataset:
        print('*** Dataset: {} ****' .format(dataset_name))
        print('N_train: {}   N_test: {}   D: {} \n' .format(y_train.size, y_test.size, x_train.shape[1]))
           
    if dataset_name != 'sarcos':
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    y_train_mean, y_train_std = y_train.mean(), y_train.std()
    y_train = (y_train - y_train_mean)/y_train_std
    y_test = (y_test - y_train_mean)/y_train_std
        
    return x_train, x_test, y_train, y_test 
 
    
def compute_rmse_nlpd(y_true, y_pred, var_test):
    mse_term = np.square(y_true - y_pred)
    nlpd = 0.5*(np.log(2.*np.pi*var_test) + mse_term/var_test)
    rmse = np.sqrt(mse_term.mean())

    return rmse, nlpd.mean()
    
    
# Get flags from the command line
def get_flags():
    flags = tf.compat.v1.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('batch_size', 2048, 'Batch size')
    flags.DEFINE_integer('num_epochs', 400, 'Number of epochs used to train DMGP/DFGP model')
    flags.DEFINE_integer('display_freq', 10, 'Display loss function value every FLAGS.display_freq epochs')
    flags.DEFINE_integer('num_splits', 1, 'Number of random data splits used - number of experiments run for a model')
    flags.DEFINE_integer('rank', 100, 'Rank r for MGP, FGP, SGPR')
    flags.DEFINE_integer('d_mgp', 5, 'Number of output dimensions for MGP\'s projection')
    flags.DEFINE_string('dataset', 'elevators', 'Dataset name')

    return FLAGS
       
  