import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, ReLU, LogSoftmax, Sequential, Identity
import torch.nn as nn
from model import GCN


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, **kwargs):
        super(MLP, self).__init__()
        self.lin = Sequential(Dropout(dropout), Linear(in_channels, hidden_channels), Dropout(dropout, inplace=True), ReLU(inplace=True), Linear(hidden_channels, out_channels), LogSoftmax(dim=1))
        self.dropout = dropout

    def reset_parameters(self):
        pass

    def forward(self, data, **kwargs):
        x = data.x
        return self.lin(x)


class APPNP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, prop, **kwargs):
        super(APPNP, self).__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.prop = prop

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, adj_t, = data.x, data.adj_t
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, adj_t, data=data)
        return F.log_softmax(x, dim=1)


# class ALTOPT(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, dropout, prop, args, **kwargs):
#         super(ALTOPT, self).__init__()
#         self.lin1 = Linear(in_channels, hidden_channels)
#         self.lin2 = Linear(hidden_channels, out_channels)
#         self.dropout = dropout
#         self.prop = prop
#         self.args = args
#         self.add_self_loops = True  # maybe false @Zhou 2003
#         self.FF = None  ## pseudo label
#         self.mlp = None
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         self.lin1.reset_parameters()
#         self.lin2.reset_parameters()
#         self.prop.reset_parameters()
#         self.mlp = None
#         self.FF = None
#
#     def propagate_update(self, data, K):
#         # return
#
#         # pass
#         if self.FF is None:
#             self.FF = self.prop.init_label(data)
#         # else:
#         ## maybe recompute mlp output here using the test mode
#         if self.mlp is None:
#             # print('no mlp')
#             zero_mlp = torch.zeros_like(self.FF)
#             self.FF = self.prop(x=zero_mlp, edge_index=data.adj_t, data=data, FF=self.FF, mode='ALTOPT')
#         else:
#             # print('with mlp')
#             self.FF = self.prop(x=self.mlp, edge_index=data.adj_t, data=data, FF=self.FF, mode='ALTOPT')
#
#     def forward(self, data):
#         x, adj_t = data.x, data.adj_t
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.lin2(x)
#         x = F.softmax(x, dim=1)
#         ## note that the difference between training and test: dropout or not
#         if not self.training:
#             ## there is no dropout in test
#             self.mlp = x.clone().detach()
#         return x

class ALTOPT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, prop, args, **kwargs):
        super(ALTOPT, self).__init__()
        num_layers = args.num_layers
        self.hidden_channels = hidden_channels
        self.lins = Sequential()
        self.lins.append(nn.Dropout(dropout))
        if num_layers > 1:
            self.lins.append(Linear(in_channels, hidden_channels))
            if args.bn:
                self.lins.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Dropout(dropout, inplace=True))
            self.lins.append(nn.ReLU(inplace=True))
            for _ in range(num_layers - 2):
                self.lins.append(Linear(hidden_channels, hidden_channels))
                if args.bn:
                    self.lins.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Dropout(dropout, inplace=True))
                self.lins.append(nn.ReLU(inplace=True))
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, out_channels))
            if args.bn:
                self.lins.append(nn.BatchNorm1d(out_channels))
        if args.tailln:
            self.lins.append(torch.nn.LayerNorm(out_channels, elementwise_affine=False))
        if args.loss == "CE":
            self.lins.append(torch.nn.LogSoftmax(dim=-1))
        self.lins.append(nn.LogSoftmax(dim=-1) if args.loss == "CE" else nn.Identity())
        self.dropout = dropout
        self.prop = prop
        self.args = args
        self.add_self_loops = True
        self.FF = None
        self.mlp = None
        self.useGCN = args.useGCN
        self.gcn = GCN(in_channels, hidden_channels, out_channels, dropout, 2, args)

    def reset_parameters(self):
        '''
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.prop.reset_parameters()
        '''
        self.gcn.reset_parameters()
        self.mlp = None
        self.FF = None


    def propagate(self, data, K=None):
        x, adj_t, = data.x, data.adj_t
        self.x = self.prop(x, adj_t, data=data, mode='CS', K=K)
        if self.FF is None:
            self.FF = self.prop.init_label(data)

    def propagate_update(self, data, K):
        if self.FF is None:
            self.FF = self.prop.init_label(data)
        if self.mlp is None:
            zero_mlp = torch.zeros_like(self.FF)
            self.FF = self.prop(x=zero_mlp, edge_index=data.adj_t, data=data, FF=self.FF, mode='ALTOPT')
        else:
            self.FF = self.prop(x=self.mlp, edge_index=data.adj_t, data=data, FF=self.FF, mode='ALTOPT')

    def forward(self, data, index=None):
        if self.FF is None:
            self.FF = self.prop.init_label(data)
        if self.useGCN:
            x = self.gcn(data)
        else:
            # x, adj_t = data.x, data.adj_t
            x = self.x
            x = self.lins(x)

        if not self.training:
            self.mlp = x.clone().detach()
        return x


class ExactALTOPT(ALTOPT):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, prop, args, **kwargs):
        super().__init__(in_channels, hidden_channels, out_channels, dropout, prop, args, **kwargs)

    def propagate_update(self, data, K):
        if self.FF is None:
            self.FF = self.prop.init_label(data)
        if self.mlp is None:
            zero_mlp = torch.zeros_like(self.FF)
            self.FF = self.prop(x=zero_mlp, edge_index=data.adj_t, data=data, FF=self.FF, mode='EXACT')
        else:
            self.FF = self.prop(x=self.mlp, edge_index=data.adj_t, data=data, FF=self.FF, mode='EXACT')
    
class AGD(ALTOPT):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, prop, args, **kwargs):
        super().__init__(in_channels, hidden_channels, out_channels, dropout, prop, args, **kwargs)

    def propagate_update(self, data, K):
        if self.FF is None:
            self.FF = self.prop.init_label(data)
        if self.mlp is None:
            zero_mlp = torch.zeros_like(self.FF)
            self.FF = self.prop(x=zero_mlp, edge_index=data.adj_t, data=data, FF=self.FF, mode='AGD')
        else:
            self.FF = self.prop(x=self.mlp, edge_index=data.adj_t, data=data, FF=self.FF, mode='AGD')