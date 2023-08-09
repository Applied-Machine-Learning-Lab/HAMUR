

import torch
from torch import nn
from torch.nn.parameter import Parameter
from ...basic.layers import LR, MLP, CrossNetwork, EmbeddingLayer


class DCN_MD(torch.nn.Module):
    """
    multi-domain Deep & Cross Network
    """

    def __init__(self, features,num_domains , n_cross_layers, mlp_params):
        super().__init__()
        self.features = features
        self.dims = sum([fea.embed_dim for fea in features])
        self.num_domains = num_domains
        self.embedding = EmbeddingLayer(features)
        self.cn = CrossNetwork(self.dims, n_cross_layers)
        self.mlp = MLP(self.dims, output_layer=False, **mlp_params)
        self.linear = LR(self.dims + mlp_params["dims"][-1])

    def forward(self, x):
        domain_id = x["domain_indicator"].clone().detach()

        embed_x = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        # mask list
        mask = []
        # out list
        out = []

        for d in range(self.num_domains):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input = embed_x
            cn_out = self.cn(domain_input)
            mlp_out = self.mlp(domain_input)
            x_stack = torch.cat([cn_out, mlp_out], dim=1)
            y = self.linear(x_stack)
            out.append(torch.sigmoid(y))

        final = torch.zeros_like(out[0])
        for d in range(self.num_domains):
            final = torch.where(mask[d].unsqueeze(1), out[d], final)
        return final.squeeze(1)


class DCN_MD_adp(torch.nn.Module):
    """
    multi-domain Deep & Cross Network with Adapter
    """

    def __init__(self, features, num_domains, n_cross_layers, k, mlp_params ,hyper_dims):
        super().__init__()
        self.features = features
        self.dims = sum([fea.embed_dim for fea in features])
        self.num_domains = num_domains
        self.embedding = EmbeddingLayer(features)
        self.cn = CrossNetwork(self.dims, n_cross_layers)
        self.mlp = MLP(self.dims, output_layer=False, **mlp_params)
        self.linear = LR(self.dims + mlp_params["dims"][-1])



        # instance represntation matrix init
        self.k = k

        # u,v initiation
        self.u = nn.ParameterList()
        self.v = nn.ParameterList()

        self.u.append(Parameter(torch.ones((mlp_params["dims"][-1], self.k)), requires_grad=True))
        self.u.append(Parameter(torch.ones((32, self.k)), requires_grad=True))

        self.v.append(Parameter(torch.ones((self.k, 32)), requires_grad=True))
        self.v.append(Parameter(torch.ones((self.k, mlp_params["dims"][-1])), requires_grad=True))

        # hyper-network initiation
        hyper_dims += [self.k * self.k]
        input_dim = self.dims
        hyper_layers = []
        for i_dim in hyper_dims:
            hyper_layers.append(nn.Linear(input_dim, i_dim))
            hyper_layers.append(nn.BatchNorm1d(i_dim))
            hyper_layers.append(nn.ReLU())
            hyper_layers.append(nn.Dropout(p=0))
            input_dim = i_dim
        self.hyper_net = nn.Sequential(*hyper_layers)

        # adapter parameters initiation
        self.b_list = nn.ParameterList()
        self.b_list.append(Parameter(torch.zeros((32)), requires_grad=True))
        self.b_list.append(Parameter(torch.zeros(mlp_params["dims"][-1]), requires_grad=True))

        self.gamma1 = nn.Parameter(torch.ones(mlp_params["dims"][-1]))
        self.bias1 = nn.Parameter(torch.zeros(mlp_params["dims"][-1]))
        self.eps = 1e-5

    def forward(self, x):
        domain_id = x["domain_indicator"].clone().detach()

        embed_x = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        # mask 存储每个batch中的domain id
        mask = []
        # out 存储
        out_l = []

        for d in range(self.num_domains):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input = embed_x

            hyper_out_full = self.hyper_net(domain_input)  # B * (k * k)
            hyper_out = hyper_out_full.reshape(-1, self.k, self.k)  # B * k * k

            cn_out = self.cn(domain_input)
            mlp_out = self.mlp(domain_input)


            # Adapter-cell
            # Adapter-layer1: Down projection
            w1 = torch.einsum('mi,bij,jn->bmn', self.u[0], hyper_out, self.v[0])
            b1 = self.b_list[0]
            tmp_out = torch.einsum('bf,bfj->bj', mlp_out, w1)
            tmp_out += b1
            # Adapter-layer2: non-linear
            tmp_out = torch.sigmoid(tmp_out)
            # Adapter-layer3: Up projection
            w2 = torch.einsum('mi,bij,jn->bmn', self.u[1], hyper_out, self.v[1])
            b2 = self.b_list[1]
            tmp_out = torch.einsum('bf,bfj->bj', tmp_out, w2)
            tmp_out += b2
            # Adapter-layer4: Domain norm
            mean = tmp_out.mean(dim=0)
            var = tmp_out.var(dim=0)
            x_norm = (tmp_out - mean) / torch.sqrt(var + self.eps)
            out = self.gamma1 * x_norm + self.bias1
            # Adapter: short-cut
            mlp_out = out + mlp_out

            x_stack = torch.cat([cn_out, mlp_out], dim=1)
            y = self.linear(x_stack)
            out_l.append(torch.sigmoid(y))

        final = torch.zeros_like(out_l[0])
        for d in range(self.num_domains):
            final = torch.where(mask[d].unsqueeze(1), out_l[d], final)
        return final.squeeze(1)
