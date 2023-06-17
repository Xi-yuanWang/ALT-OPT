from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch_sparse
from torch_sparse import SparseTensor, matmul
import numpy as np
import math
import cupy as cp
import cupy.sparse as cpsp
import cupy.sparse.linalg as cpsplg
from util import cg



class Propagation(MessagePassing):
    r"""The elastive message passing layer from 
    the paper "Elastic Graph Neural Networks", ICML 2021
    """

    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, 
                 K: int, 
                 mode: str,
                 lambda1: float = None,
                 lambda2: float = None,
                 alpha: float = None,
                 L21: bool = True,
                 dropout: float = 0,
                 cached: bool = False,
                 add_self_loops: bool = True,
                 normalize: bool = True,
                 args = None,
                 **kwargs):

        super(Propagation, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.mode = mode
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.L21 = L21
        self.alpha = alpha
        self.dropout = dropout
        self.cached = cached
        
        assert add_self_loops == True and normalize == True, ''
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_adj_t = None
        self.args = args
        self.label = None
        self.num_class = args.num_class


    def reset_parameters(self):
        self._cached_adj_t = None
        self.label = None

    def forward(self, x: Tensor, 
                edge_index: Adj, 
                edge_weight: OptTensor = None, 
                data=None,
                FF=None,
                mode=None,
                post_step=None, alpha=None, K=None) -> Tensor:
        if K is None:
            K = self.K
        if self.K <= 0: return x

        assert isinstance(edge_index, SparseTensor), "Only support SparseTensor now"
        assert edge_weight is None, "edge_weight is not supported yet, but it can be extented to weighted case"

        if self.normalize:
            cache = self._cached_adj_t
            if cache is None:
                edge_index = gcn_norm(edge_index, add_self_loops=self.add_self_loops)
                if self.cached:
                    self._cached_adj_t = edge_index
            else:
                edge_index = cache

        hh = x
        mode = self.mode if mode is None else mode
        if alpha is None:
            alpha = self.alpha

        if mode == 'APPNP':
            x = self.appnp_forward(x=x, hh=hh, edge_index=edge_index, K=K, alpha=alpha)
        elif mode == 'ALTOPT':
            x = self.apt_forward(mlp=x, FF=FF, edge_index=edge_index, K=K, alpha=alpha, data=data)
        elif mode == 'CS':
            x = self.label_forward(x=x, edge_index=edge_index, K=K, alpha=alpha, post_step=post_step,
                                   edge_weight=edge_weight)
        elif mode == 'ORTGNN':
            x = self.ort_forward(x=x, edge_index=edge_index, K=K, alpha=alpha, data=data)
        elif mode == "EXACT":
            x = self.exact_forward(mlp=x, FF=FF, edge_index=edge_index, K=K, alpha=alpha, data=data)
        elif mode == "AGD":
            x = self.agd_forward(mlp=x, FF=FF, edge_index=edge_index, K=K, alpha=alpha, data=data)
        else:
            raise ValueError('wrong propagate mode')
        return x

    def init_label(self, data, nodes=None, classes=None):
        mask = data.train_mask
        nodes = data.x.shape[0]
        classes = data.y.max() + 1
        label = torch.zeros(nodes, classes).cuda()
        label[mask, data.y[mask]] = 1
        return label

    def apt_forward(self, mlp, FF, edge_index, K, alpha, data):
        lambda1 = self.args.lambda1
        lambda2 = self.args.lambda2
        wd = self.args.Fwd
        if not torch.is_tensor(self.label):
            self.label = self.init_label(data)
            print('init label')
        label = self.label
        mask = data.train_mask

        for k in range(K):
            AF = self.propagate(edge_index, x=FF, edge_weight=None, size=None)
            if self.args.loss == 'CE':
                FF[mask] = lambda1/(2*(1+lambda2)) * mlp[mask] + 1/(1+lambda2) * AF[mask] + lambda2/(1+lambda2)*label[mask] - wd*FF[mask]
                FF[~mask] = lambda1/2 * mlp[~mask] + AF[~mask] - wd*FF[~mask]
            else:
                FF[mask] = 1/(lambda1+lambda2+1) * AF[mask] + lambda1/(lambda1+lambda2+1) * mlp[mask] + lambda2/(lambda1+lambda2+1) * label[mask]  ## for labeled nodes
                FF[~mask] = 1/(lambda1+1) * AF[~mask] + lambda1/(lambda1+1) * mlp[~mask] ## for unlabeled nodes
        return FF
    

    def exact_forward(self, mlp, FF, edge_index: SparseTensor, K, alpha, data):
        wd = self.args.Fwd
        lambda1 = self.args.lambda1
        lambda2 = self.args.lambda2
        onlyy = self.args.onlyy
        usecg = self.args.usecg
        softmaxF = self.args.softmaxF
        if not torch.is_tensor(self.label):
            self.label = self.init_label(data)
            print('init label')
        label = self.label
        mask = data.train_mask

        if getattr(data, "Leftmat", None) is None:
            N = data.num_nodes
            rowptr, col, val = edge_index.csr()
            
            rowptr, col, val = cp.asarray(rowptr), cp.asarray(col), cp.asarray(val)
            if self.args.loss == 'CE':
                diagterm = torch.ones(N, device=label.device) + wd
                diagterm[mask] += lambda2
            else:
                diagterm = torch.empty(N, device=label.device).fill_(lambda1+1+wd)
                diagterm[mask] += lambda2
            Leftmat = cpsp.csr_matrix((-val, col, rowptr), dtype=cp.float32, shape=(N, N))
            Leftmat.setdiag(cp.asarray(diagterm))
            setattr(data, "Leftmat", Leftmat)
        Leftmat = data.Leftmat
        if getattr(data, "plabel", None) is None:
            plabel = torch.as_tensor(cpsplg.spsolve(Leftmat, cp.asarray(label)), device=label.device)
            setattr(data, "plabel", plabel)
        if onlyy:
            FF = data.plabel
        else:
            if self.args.loss == 'CE':
                Rightmat = (lambda1/2)*mlp
            else:
                Rightmat = lambda1*mlp
            if usecg:
                FF = cg(Leftmat, cp.asarray(Rightmat), cp.asarray(FF), K)
            else:
                FF = cpsplg.spsolve(Leftmat, cp.asarray(Rightmat)) 
            FF = torch.as_tensor(FF, device=label.device) + data.plabel
        return FF


    def agd_forward(self, mlp, FF, edge_index: SparseTensor, K, alpha, data):
        lambda1 = self.args.lambda1
        lambda2 = self.args.lambda2
        wd = self.args.Fwd

        if not torch.is_tensor(self.label):
            self.label = self.init_label(data)
            print('init label')
        label = self.label
        mask = data.train_mask

        if getattr(self, "agdcoeff", None) is None:
            thetas = [1]
            ttheta = 1
            coeff = []
            for i in range(1, K+1):
                ttheta = ((ttheta**4 + 4*ttheta**2)**0.5-ttheta**2)/2
                coeff.append(ttheta*(1-thetas[-1])/thetas[-1])
                thetas.append(ttheta)
            setattr(self, "agdcoeff", coeff)
        agdcoeff = self.agdcoeff

        if getattr(data, "diagterm", None) is None:
            N = data.num_nodes
            if self.args.loss == 'CE':
                diagterm = torch.empty(N, device=label.device).fill_(lambda1+2+wd)
                diagterm[mask] += lambda2
            else:
                diagterm = torch.empty(N, device=label.device).fill_(2+wd)
                diagterm[mask] += lambda2
            setattr(data, "diagterm", 1/diagterm.reshape(-1, 1))

        diagterm = data.diagterm

        if self.args.loss == 'CE':
            Rightmat = (lambda1/2)*mlp + lambda2*label
        else:
            Rightmat = lambda1*mlp + lambda2*label
        biasterm = diagterm*Rightmat
        
        G = FF        
        FF = biasterm + diagterm * (edge_index@G)
        deltaF = FF-G
        for i in range(1, K):
            G = FF + agdcoeff[i] * deltaF
            nF = biasterm + diagterm * (edge_index@G)
            deltaF, FF = nF-FF, nF
        return FF

    def appnp_forward(self, x, hh, edge_index, K, alpha):
        for k in range(K):
            Ax = self.propagate(edge_index, x=x, edge_weight=None, size=None)
            x = alpha * hh + (1 - alpha) * Ax
        return x

    def label_forward(self, x, edge_index, K, alpha, post_step, edge_weight):
        out = x
        res = (1-alpha) * out
        for k in range(K):
            out = self.propagate(edge_index, x=out, edge_weight=edge_weight, size=None)
            out.mul_(alpha).add_(res)
            if post_step != None:
                out = post_step(out)
        return out

    def ort_forward(self, x, edge_index, K, alpha, data):
        lambda1 = self.args.lambda1
        lambda2 = self.args.lambda2
        out = x
        res = 1/(1+lambda1-2*lambda2) * out
        for k in range(K):
            AF = self.propagate(edge_index, x=out, edge_weight=None, size=None)
            FTF = torch.mm(out.T, out)
            FFTF = torch.mm(out, FTF)
            out = lambda1 / (1+lambda1-2*lambda2) * AF + res - 2*lambda2/(1+lambda1-2*lambda2)*FFTF
        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={}, mode={}, lambda1={}, lambda2={}, L21={}, alpha={})'.format(
            self.__class__.__name__, self.K, self.mode, self.lambda1, self.lambda2, self.L21, self.alpha)
