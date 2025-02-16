'''
***Closely based on code for TVAE by Matthew J. Vowels***
https://github.com/matthewvowels1/TVAE_release
See also the paper:
Vowels, M. J., Camgoz, N. C., & Bowden, R. (2021). Targeted VAE: Variational 
and targeted learning for causal inference. https://arxiv.org/pdf/2009.13472
'''
'''
***Closely based on code for NetConf by Ruocheng Guo***
https://github.com/rguo12/network-deconfounder-wsdm20
See also the paper:
Guo, R., Li, J., & Liu, H. (2020). Learning individual causal effects from 
networked observational data. https://dl.acm.org/doi/pdf/10.1145/3336191.3371816
'''
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.nn import PyroModule
from pyro.optim import ClippedAdam
from pyro.util import torch_isnan,check_if_enumerated, warn_if_nan
from pyro.distributions.util import is_identically_zero
from tndvga_guide_model import Model, Guide
from pyro.infer.enum import get_importance_trace
from pyro.infer.util import (
    MultiFrameTensor,
    get_plate_stacks,
    torch_item,
)
import torch.nn.functional as F
from hsic import HSIC,RbfHSIC
import numpy as np
from torch.nn.utils import clip_grad_norm_

logger = logging.getLogger(__name__)

def _compute_log_r(model_trace, guide_trace):
    log_r = MultiFrameTensor()
    stacks = get_plate_stacks(model_trace)
    for name, model_site in model_trace.nodes.items():
        if model_site["type"] == "sample":
            log_r_term = model_site["log_prob"]
            if not model_site["is_observed"]:
                log_r_term = log_r_term - guide_trace.nodes[name]["log_prob"]
            log_r.add((stacks[name], log_r_term.detach()))
    return log_r

class TraceCausalEffect_ELBO(Trace_ELBO):
    def _differentiable_loss_particle(self, model_trace, guide_trace):
        blocked_names = [name for name, site in guide_trace.nodes.items()
                         if site["type"] == "sample" and site["is_observed"]]
        blocked_guide_trace = guide_trace.copy()
        for name in blocked_names:
            del blocked_guide_trace.nodes[name]
        elbo_particle = 0
        surrogate_elbo_particle = 0
        log_r = None
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample":
                elbo_particle = elbo_particle + torch_item(torch.sum(site["log_prob"][self.idx]))
                surrogate_elbo_particle = surrogate_elbo_particle + torch.sum(site["log_prob"][self.idx])
            
        for name, site in blocked_guide_trace.nodes.items():
            if site["type"] == "sample":
                log_prob, score_function_term, entropy_term = site["score_parts"]

                elbo_particle = elbo_particle - torch_item(torch.sum(site["log_prob"][self.idx]))
                if not is_identically_zero(entropy_term):
                    surrogate_elbo_particle = (
                        surrogate_elbo_particle - entropy_term[self.idx].sum()
                    )

                if not is_identically_zero(score_function_term):
                    if log_r is None:
                        log_r = _compute_log_r(model_trace, blocked_guide_trace)
                    site = log_r.sum_to(site["cond_indep_stack"])
                    surrogate_elbo_particle = (
                        surrogate_elbo_particle + (site * score_function_term)[self.idx].sum()
                    )
        loss=-elbo_particle
        surrogate_loss=-surrogate_elbo_particle
        
        
        for name in blocked_names:
            log_q = guide_trace.nodes[name]["log_prob"][self.idx].sum()
            loss = loss - 100* torch_item(log_q)
            surrogate_loss = surrogate_loss - 100* log_q
        return loss, surrogate_loss
    
    @torch.no_grad()
    def loss(self, model, guide, *args, **kwargs):
        return torch_item(self.differentiable_loss(model, guide, *args, **kwargs))

    
class TNDVGA(nn.Module):
    def __init__(self, 
                 feature_dim, 
                 continuous_dim, 
                 outcome_dist="normal", 
                 latent_dim_o=20,
                 latent_dim_c=20, 
                 latent_dim_t=20, 
                 latent_dim_y=20, 
                 hidden_dim=200, 
                 num_layers=3, 
                 num_samples=100):
        super().__init__()
        config = dict(feature_dim=feature_dim, 
                      latent_dim_c=latent_dim_c,
                      latent_dim_o=latent_dim_o,
                      latent_dim_t = latent_dim_t, 
                      latent_dim_y = latent_dim_y,
                      hidden_dim=hidden_dim, 
                      num_layers=num_layers, 
                      continuous_dim = continuous_dim, 
                      num_samples=num_samples)
        config["outcome_dist"] = outcome_dist
        
        self.feature_dim = feature_dim
        self.num_samples = num_samples
        self.model = Model(config)
        self.guide = Guide(config)
        self.cuda()
        
    def fit(self, x, adj, t, y, idx_train, 
            func="median",
            num_epochs=100,
            learning_rate=1e-3,
            learning_rate_decay=0.1,
            weight_decay=1e-4,
            dist_weight=1e-4,
            hsic_weight=100):
        
        assert x.dim() == 2 and x.size(-1) == self.feature_dim
        assert t.shape == x.shape[:1]
        assert y.shape == y.shape[:1]
        
        num_steps = num_epochs
        
        inf_all = [param
                        for name, param in list(self.guide.named_parameters())
                        if param.requires_grad]
        gen_all = [param
                        for name, param in list(self.model.named_parameters())
                        if param.requires_grad]
        main_params = list(gen_all) + list(inf_all)
        
     
        optim_main = torch.optim.Adam([{"params": main_params, "lr": learning_rate,
                             "weight_decay": weight_decay,
                             "lrd": learning_rate_decay ** (1 / num_steps)}])
        
        loss_fn = TraceCausalEffect_ELBO(idx=idx_train).differentiable_loss
        total_losses = []
        for epoch in range(num_epochs):
            zo,zc,zt,zy=self.guide.z_loc(x,adj)
            zy_t1, zy_t0 = zy[idx_train][(t[idx_train] > 0).nonzero()], zy[idx_train][(t[idx_train] < 1).nonzero()]
            zo_=zo[idx_train]
            zc_=zc[idx_train]
            zt_=zt[idx_train]
            zy_=zy[idx_train]
            
            hsic = RbfHSIC(sigma_x=1, sigma_y=1, algorithm='unbiased',
                 reduction=None)
            indep_reg = hsic(zo_,zc_)+hsic(zo_,zt_)+hsic(zo_,zy_)+hsic(zc_,zt_)+hsic(zc_,zy_)+hsic(zt_,zy_)
            main_loss_train = loss_fn(self.model, self.guide, x, adj, t, y, size=x.shape[0])/ x.shape[0]+dist_weight*self.wasserstein(zy_t1, zy_t0, cuda=True)[0]+hsic_weight*indep_reg
            
            optim_main.zero_grad()
            main_loss_train.backward()
            clip_grad_norm_(main_params, max_norm=1000.0)
            optim_main.step()
            
            total_loss = main_loss_train / x.shape[0]
            print("step {: >5d} loss = {:0.6g}".format(epoch, total_loss))
            assert not torch_isnan(total_loss)
            total_losses.append(total_loss)
            
        return total_losses
        
    def wasserstein(self,x,y,p=0.5,lam=1,its=20,sq=False,backpropT=False,cuda=False):
        """return W dist between x and y"""
        '''distance matrix M'''
        nx = x.shape[0]
        ny = y.shape[0]

        x = x.squeeze()
        y = y.squeeze()


        M = self.pdist(x,y) 

        '''estimate lambda and delta'''
        M_mean = torch.mean(M)
        M_drop = F.dropout(M,10.0/(nx*ny))
        delta = torch.max(M_drop).detach()
        eff_lam = (lam/M_mean).detach()

        '''compute new distance matrix'''
        Mt = M
        row = delta*torch.ones(M[0:1,:].shape)
        col = torch.cat([delta*torch.ones(M[:,0:1].shape),torch.zeros((1,1))],0)
        if cuda:
            row = row.cuda()
            col = col.cuda()
        Mt = torch.cat([M,row],0)
        Mt = torch.cat([Mt,col],1)

        '''compute marginal'''
        a = torch.cat([p*torch.ones((nx,1))/nx,(1-p)*torch.ones((1,1))],0)
        b = torch.cat([(1-p)*torch.ones((ny,1))/ny, p*torch.ones((1,1))],0)

        '''compute kernel'''
        Mlam = eff_lam * Mt
        temp_term = torch.ones(1)*1e-6
        if cuda:
            temp_term = temp_term.cuda()
            a = a.cuda()
            b = b.cuda()
        K = torch.exp(-Mlam) + temp_term
        U = K * Mt
        ainvK = K/a

        u = a

        for i in range(its):
            u = 1.0/(ainvK.matmul(b/torch.t(torch.t(u).matmul(K))))
            if cuda:
                u = u.cuda()
        v = b/(torch.t(torch.t(u).matmul(K)))
        if cuda:
            v = v.cuda()

        upper_t = u*(torch.t(v)*K).detach()

        E = upper_t*Mt
        D = 2*torch.sum(E)

        if cuda:
            D = D.cuda()

        return D, Mlam

    def pdist(self,sample_1, sample_2, norm=2, eps=1e-5):
        n_1, n_2 = sample_1.size(0), sample_2.size(0)
        norm = float(norm)
        if norm == 2.:
            norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
            norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
            norms = (norms_1.expand(n_1, n_2) +
                     norms_2.transpose(0, 1).expand(n_1, n_2))
            distances_squared = norms - 2 * sample_1.mm(sample_2.t())
            return torch.sqrt(eps + torch.abs(distances_squared))
        else:
            dim = sample_1.size(1)
            expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
            expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
            differences = torch.abs(expanded_1 - expanded_2) ** norm
            inner = torch.sum(differences, dim=2, keepdim=False)
            return (eps + inner) ** (1. / norm)

    @torch.no_grad()
    def ite(self, x, adj, ym, ys, idx_train=None, idx_val=None, idx_test=None, num_samples=None, batch_size=None):
        if num_samples is None:
            num_samples = self.num_samples
        if not torch._C._get_tracing_state():
            assert x.dim() == 2 and x.size(-1) == self.feature_dim

        dataloader = [x] if batch_size is None else DataLoader(x, batch_size=batch_size)
        logger.info("Evaluating {} minibatches".format(len(dataloader)))
        result_ite_train = []
        result_ate_train = []
        result_ite_test = []
        result_ate_test = []
        result_ite_val = []
        result_ate_val = []
        for x in dataloader:
            with pyro.plate("num_particles", num_samples, dim=-2):
                with poutine.trace() as tr, poutine.block(hide=["y", "t"]):
                    self.guide(x, adj)
                with poutine.do(data=dict(t=torch.zeros(()))):
                    y0 = poutine.replay(self.model.y_mean, tr.trace)(x, adj) * ys + ym
                with poutine.do(data=dict(t=torch.ones(()))):
                    y1 = poutine.replay(self.model.y_mean, tr.trace)(x, adj) * ys + ym
            ite = (y1 - y0).mean(0)
            if not torch._C._get_tracing_state():
                logger.debug("batch ate = {:0.6g}".format(ite.mean()))
            ite_train = ite[idx_train]
            ite_val = ite[idx_val]
            ite_test = ite[idx_test]
            ate_train = ite_train.mean()
            ate_val = ite_val.mean()
            ate_test = ite_test.mean()
            
            result_ite_train.append(ite_train)
            result_ate_train.append(ate_train)
            result_ite_val.append(ite_val)
            result_ate_val.append(ate_val)
            result_ite_test.append(ite_test)
            result_ate_test.append(ate_test)
        return torch.cat(result_ite_train),torch.cat(result_ite_val),torch.cat(result_ite_test),result_ate_train[0],result_ate_val[0],result_ate_test[0]
    
class model_param_init(nn.Module):
    def __init__(self, model):
        super().__init__()
        assert isinstance(model, nn.Module), 'model not a class nn.Module'
        self.net = model
        self.initParam()
        
    def initParam(self):
        for param in self.net.parameters():
            nn.init.xavier_normal_(param.unsqueeze(0), gain=1) 

