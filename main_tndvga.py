 import argparse
import pandas as pd
import torch
import utils
import pyro
from tndvga_wrapper import TNDVGA,model_param_init
import numpy as np
import csv
import random
import os


def main(args,i_exp):
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    X, A, T, Y1, Y0 = utils.load_data(args.path, name=args.dataset, original_X=False, exp_id=str(i_exp), extra_str=args.extrastr) 
    n = X.shape[0]
    n_train = int(n * args.tr)
    n_test = int(n * 0.2)

    idx = np.random.permutation(n)
    idx_train, idx_test, idx_val = idx[:n_train], idx[n_train:n_train+n_test], idx[n_train+n_test:]

    X = utils.normalize(X) 

    X = X.todense()
    X = Tensor(X)
    Y1 = Tensor(np.squeeze(Y1))
    Y0 = Tensor(np.squeeze(Y0))
    T = Tensor(np.squeeze(T))
    A = utils.sparse_mx_to_torch_sparse_tensor(A,cuda=True)

    idx_train = LongTensor(idx_train)
    idx_val = LongTensor(idx_val)
    idx_test = LongTensor(idx_test)

    YF = torch.where(T>0,Y1,Y0)
    ym, ys = torch.mean(YF[idx_train]), torch.std(YF[idx_train])
    YF_=(YF - ym) / ys

    pyro.clear_param_store()
    contfeats =  [i for i in range(X.shape[1])]
    tndvga = TNDVGA(feature_dim=X.shape[1], continuous_dim= contfeats,outcome_dist='normal',
                  latent_dim_o=args.latent_dim_o, latent_dim_c=args.latent_dim_c, latent_dim_t=args.latent_dim_t,
                  latent_dim_y=args.latent_dim_y,
                  hidden_dim=args.hidden_dim,
                  num_layers=args.num_layers,
                  num_samples=100)                                                                                                                           
    model_param_init(tndvga)
    tndvga.fit(X, A, T, YF_, idx_train,
              func=args.func,
              num_epochs=args.num_epochs,
              learning_rate=args.lr,
              learning_rate_decay=args.lrd, weight_decay=args.weight_decay,dist_weight=args.dist_weight,hsic_weight=args.hsic_weight)


    est_ite_train,est_ite_val,est_ite_test, est_ate_train,est_ate_val,est_ate_test= tndvga.ite(X, A, ym, ys,idx_train,idx_val,idx_test)
    
    true_ite_train = (Y1 - Y0)[idx_train]
    true_ite_val = (Y1 - Y0)[idx_val]
    true_ite_test = (Y1 - Y0)[idx_test]

    loss = torch.nn.MSELoss().cuda()
    pehe_train = torch.sqrt(loss(est_ite_train,true_ite_train))
    eate_train = torch.abs(torch.mean(true_ite_train)-est_ate_train)
    pehe_val = torch.sqrt(loss(est_ite_val,true_ite_val))
    eate_val = torch.abs(torch.mean(true_ite_val)-est_ate_val)
    pehe_test = torch.sqrt(loss(est_ite_test,true_ite_test))
    eate_test = torch.abs(torch.mean(true_ite_test)-est_ate_test)
    
    return pehe_train, eate_train, pehe_val, eate_val, pehe_test, eate_test, torch.mean(true_ite_train), est_ate_train.cpu()

if __name__ == "__main__":   
    parser = argparse.ArgumentParser(description="TNDVGA")
    parser.add_argument('--dataset', type=str, default='Flickr')
    parser.add_argument('--path', type=str, default='./datasets/')
    parser.add_argument('--func', type=str, default='mean')

    parser.add_argument("--latent_dim_o", default=50, type=int)
    parser.add_argument("--latent_dim_c", default=50, type=int)
    parser.add_argument("--latent_dim_t", default=50, type=int)
    parser.add_argument("--latent_dim_y", default=50, type=int)
    parser.add_argument("--hidden-dim", default=500, type=int)
    parser.add_argument("--num-layers", default=2, type=int)
    parser.add_argument("--num_epochs", "--num_epochs", default=1000, type=int)
    parser.add_argument("--lr", "--learning-rate", default=3e-4, type=float)
    parser.add_argument("--lrd", "--learning-rate-decay", default=0.01, type=float)
    parser.add_argument("--weight-decay", default=5e-5, type=float)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument('--extrastr', type=str, default='2')
    parser.add_argument('--seed', type=int, default=1000, help='Random seed.')
    parser.add_argument('--dropout', type=float, default=0.1,
                            help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dist_weight', type=float, default=0.1,
                            help='trade-off of representation balancing.')
    parser.add_argument('--hsic_weight', type=float, default=1,
                            help='trade-off of independence.')
    parser.add_argument('--clip', type=float, default=100.,
                            help='gradient clipping')
    parser.add_argument('--tr', type=float, default=0.6)
    args = parser.parse_args()

    Tensor = torch.cuda.FloatTensor 
    LongTensor =torch.cuda.LongTensor

    dist_weight = Tensor([args.dist_weight])

    pehe_train = np.zeros(1)
    eate_train = np.zeros(1)
    pehe_val = np.zeros(1)
    eate_val = np.zeros(1)
    pehe_test = np.zeros(1)
    eate_test = np.zeros(1)
       
    i_exp=1
    pehe_train, eate_train, pehe_val, eate_val, pehe_test, eate_test, ate, ate_hat = main(args,i_exp)
    print('pehe_train:{:.4f}'.format(pehe_train))
    print('eate_train:{:.4f}'.format(eate_train))
    print('pehe_val:{:.4f}'.format(pehe_val))
    print('eate_val:{:.4f}'.format(eate_val))
    print('pehe_test:{:.4f}'.format(pehe_test))
    print('eate_test:{:.4f}'.format(eate_test))
    print('true ate train:{:.4f}'.format(ate), 'pred ate:{:.2f}'.format(ate_hat))


    
             
            


