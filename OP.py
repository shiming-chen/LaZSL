import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

class OP():
    def __init__(self, max_iter, M, N, n_cls, b):
        super(OP, self).__init__()
        self.max_iter= max_iter
        self.M=M
        self.N=N
        self.n_cls=n_cls
        self.b=b

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

        return T
    def get_OP_distence(self,image_features,text_features):
        sim = torch.einsum('mbd,ncd->mnbc', image_features, text_features).contiguous()
        sim = sim.view(self.M, self.N, self.b * self.n_cls)
        sim = sim.permute(2, 0, 1)
        wdist = 1.0 - sim
        xx = torch.zeros(self.b * self.n_cls, self.M, dtype=sim.dtype, device=sim.device).fill_(1. / self.M)
        yy = torch.zeros(self.b * self.n_cls, self.N, dtype=sim.dtype, device=sim.device).fill_(1. / self.N)

        with torch.no_grad():
            KK = torch.exp(-wdist / self.eps)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return None

        sim_op = torch.sum(T * sim, dim=(1, 2))
        sim_op = sim_op.contiguous().view(self.b, self.n_cls)
        return sim_op

class OP_d():
    def __init__(self, max_iter,gama,alpha,constrain_type='const'):
        super(OP_d, self).__init__()
        self.max_iter= max_iter
        #self.M=M
        self.n_cls= 1
        self.eps = 0.1
        self.gama= torch.tensor(gama,dtype=torch.half)
        self.zero= torch.tensor(-10,dtype=torch.half)
        self.constrain_type=constrain_type #['patch','att','const']
        self.alpha=alpha
       # self.b=b

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

        return T
    def get_OP_distence(self,image_features,text_features,sim_rg=None,is_constrain=False,is_cost_global=False,is_sim_global=False,is_T=False):
        device=image_features.device
        self.zero.to(device)
        self.N=text_features.shape[0]
        self.b=image_features.shape[1]
        self.M = image_features.shape[0]
        sim = torch.einsum('mbd,nd->mnb', image_features, text_features).contiguous()
        sim = sim.view(self.M, self.N, self.b * self.n_cls)
        sim = sim.permute(2, 0, 1)
        if is_cost_global:
            global_sim=sim[:,0,:].unsqueeze(1)
            region_sim=sim[:,1:,:]
            sim_global=(1-self.alpha)*global_sim + (self.alpha * region_sim)
            sim=region_sim
            self.M = sim_global.shape[1]


        if self.constrain_type=='patch':
            gama = torch.mean(sim, dim=(1, 2), keepdim=True)
        elif self.constrain_type=='att':
            gama = torch.mean(sim, dim=2, keepdim=True)
        mask_att=sim< gama
        wdist = 1.0 - sim
        if is_cost_global:
            wdist = 1.0 - sim_global

        xx = torch.zeros(self.b * self.n_cls, self.M, dtype=sim.dtype, device=sim.device).fill_(1. / self.M)
        yy = torch.zeros(self.b * self.n_cls, self.N, dtype=sim.dtype, device=sim.device).fill_(1. / self.N)
        if sim_rg is not None:
            sim_mean=torch.mean(sim_rg,dim=1, keepdim=True)+self.gama
            mask=sim_rg<sim_mean
            if mask.shape[1] != xx.shape[1]:
                mask_g=torch.zeros((mask.shape[0],1),dtype=torch.bool,device=sim_mean.device)
                mask=torch.cat((mask_g,mask),dim=1)

            neg_mask=~mask
            xx[mask]=0
            x= torch.ones(self.b, dtype=sim.dtype,device=sim.device)/ torch.count_nonzero(xx, dim=1).to(sim.device)

            xx=torch.where(neg_mask,x.unsqueeze(-1),xx)

        with torch.no_grad():

            KK=-wdist / self.eps
            if is_constrain:
                KK[mask_att] = self.zero
            KK = torch.exp(KK)


            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return None
        if is_constrain:
            sim[mask_att]=0
        sim_op = torch.sum(T * sim, dim=(1, 2))
        if is_sim_global:
            sim_op = torch.sum(T * sim_global, dim=(1, 2))



        sim_op = sim_op.contiguous().view(self.b)
        if is_T:
            return sim_op,T,sim_global
        else:
            return sim_op






