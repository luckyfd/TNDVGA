'''
***Closely based on code for NetConf by Ruocheng Guo***
https://github.com/rguo12/network-deconfounder-wsdm20
See also the paper:
Guo, R., Li, J., & Liu, H. (2020). Learning individual causal effects from 
networked observational data. https://dl.acm.org/doi/pdf/10.1145/3336191.3371816
'''
import numpy as np
import scipy.io as sio
import scipy.sparse as sp

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from scipy import interp

def sparse_mx_to_torch_sparse_tensor(sparse_mx,cuda=False):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    if cuda:
        sparse_tensor = sparse_tensor.cuda()
    return sparse_tensor

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def load_data(path, name='BlogCatalog',exp_id='0',original_X = False, extra_str=""):
    data = sio.loadmat(path+name+extra_str+'/'+name+exp_id+'.mat')
    A = data['Network'] 

    if not original_X:
        X = data['X_100']
    else:
        X = data['Attributes']

    Y1 = data['Y1']
    Y0 = data['Y0']
    T = data['T']

    return X, A, T, Y1, Y0


