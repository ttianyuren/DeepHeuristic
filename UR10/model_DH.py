import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import dgl
from dgl import DGLGraph
from gcn_layer import GCNLayer, GraphSAGELayer

BATCH_NORM = True
GC_BIAS = True


class SGNN(nn.Module):

    def __init__(self, icosphere, min_level, gconv_type, gconv_dims, gconv_depths, outblock_hidden_dims,
                 in_dim, out_dim, pool_type="none", dropout=0, residual=True, gcn_norm=False, debug=False):
        super(SGNN, self).__init__()
        assert min_level <= icosphere.level
        self.icosphere = icosphere
        self.max_level = icosphere.level
        self.min_level = min_level
        self.gconv_dims = gconv_dims
        self.gconv_depths = gconv_depths
        self.pool_type = pool_type
        self.n_conv_blocks = len(self.gconv_depths)
        self.activation = F.relu
        self.dropout = dropout

        self.gconv_blocks = nn.ModuleList()
        self.pool_blocks = nn.ModuleList()

        gconv_in_dim = in_dim
        for i in range(self.max_level - self.min_level):
            downsample_lvl = self.max_level - (i + 1)  # 3 -> 2 -> 1
            gconv_out_dim = self.gconv_dims[i]
            gconv_h_dim = gconv_out_dim
            gconv_depth = gconv_depths[i]
            g = self.icosphere.graphs_by_level[self.max_level - i]
            conv_block = ConvBlock(g, gconv_type, gconv_in_dim, gconv_h_dim, gconv_out_dim, gconv_depth,
                                   self.activation, gcn_norm, residual, 0, debug=debug)
            self.gconv_blocks.append(conv_block)

            # pooling and downsampling vertices
            pool_block = Pool(downsample_lvl,
                              self.icosphere.base_pool_inds[downsample_lvl],
                              self.icosphere.rest_pool_inds[downsample_lvl],
                              pool_type=self.pool_type)
            self.pool_blocks.append(pool_block)

            gconv_in_dim = self.gconv_dims[i]

        # base level convolutions
        idx_last = self.max_level - self.min_level
        g_last = self.icosphere.graphs_by_level[1]
        gconv_out_dim = self.gconv_dims[idx_last]
        gconv_h_dim = gconv_out_dim
        gconv_depth = gconv_depths[idx_last]
        conv_block = ConvBlock(g_last, gconv_type, gconv_in_dim, gconv_h_dim, gconv_out_dim, gconv_depth,
                               self.activation, gcn_norm, residual, 0, debug=debug)
        self.gconv_blocks.append(conv_block)

        output_in_dim = gconv_out_dim
        self.output_block = OutputBlock(output_in_dim, outblock_hidden_dims, out_dim, self.dropout, debug=debug)

    def forward(self, x):
        """
        Input: [n_batch, n_vertex, n_feats]
        """
        is_test = not self.training
        n_batch, n_vertex, n_features = x.size()
        prev_embeddings = []
        x = x.permute((1, 0, 2))  # [n_vertex, n_batch, n_features]
        # convolutions up to base level
        for i in range(self.max_level - self.min_level):
            x = self.gconv_blocks[i](x)
            x = self.pool_blocks[i](x)

        idx_last = self.max_level - self.min_level
        x = self.gconv_blocks[idx_last](x)

        x = x.permute((1, 0, 2))  # [n_batch, n_vertex, n_features]
        x = self.output_block(x)
        # print('after: ', x.shape)
        return x


class ConvBlock(nn.Module):

    def __init__(self,
                 graph,
                 gconv_type,
                 in_feats,
                 n_hidden,
                 out_feats,
                 n_layers,
                 activation,
                 gcn_norm,
                 residual,
                 dropout,
                 debug=False):
        """
        n_layers: # hidden layers
        """
        super(ConvBlock, self).__init__()
        self.debug = debug
        self.gconv_type = gconv_type
        graph = dgl.from_networkx(graph)
        if gconv_type == "gcn":
            graph = dgl.add_self_loop(graph)
        self.g = graph.to(torch.device('cuda'))

        self.conv_layers = nn.ModuleList()
        self.in_feats = in_feats  # e.g. = 8
        self.out_feats = out_feats  # e.g. = 32
        curr_in_feats = in_feats
        curr_out_feats = None
        for i in range(n_layers):
            curr_out_feats = n_hidden
            if self.gconv_type == "gcn":
                self.conv_layers.append(GCNLayer(self.g, curr_in_feats, curr_out_feats, activation, normalize=gcn_norm,
                                                 batch_norm=BATCH_NORM, dropout=dropout, residual=residual,
                                                 bias=GC_BIAS))
            elif self.gconv_type == "graphsage":
                self.conv_layers.append(
                    GraphSAGELayer(self.g, curr_in_feats, curr_out_feats, activation, batch_norm=BATCH_NORM,
                                   dropout=dropout, residual=residual, bias=GC_BIAS))

            curr_in_feats = curr_out_feats

        # last convolution layer
        curr_out_feats = out_feats
        if self.gconv_type == "gcn":
            self.conv_layers.append(
                GCNLayer(self.g, curr_in_feats, curr_out_feats, activation, normalize=gcn_norm, batch_norm=BATCH_NORM,
                         dropout=dropout, residual=residual, bias=GC_BIAS))
        elif self.gconv_type == "graphsage":
            self.conv_layers.append(
                GraphSAGELayer(self.g, curr_in_feats, curr_out_feats, activation, batch_norm=BATCH_NORM,
                               dropout=dropout, residual=residual, bias=GC_BIAS))

    def forward(self, features):
        h = features
        for i, conv_layer in enumerate(self.conv_layers):
            h = conv_layer(h)

        return h

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, conv_layers=\
                \n{})'.format(self.__class__.__name__,
                              self.in_feats,
                              self.out_feats,
                              self.conv_layers)


class Pool(nn.Module):
    """
    Pool and downsample input to match specified refinement mesh level
    Input: [V_l, n_batch, n_features]
    Input: [V_{l-1}, n_batch, n_features]
    """

    def __init__(self, level, base_pool_idx, rest_pool_idx, pool_type=None):
        super(Pool, self).__init__()
        self.level = level
        self.base_pool_idx = base_pool_idx  # indicates how to pool around the 12 base vertices
        self.rest_pool_idx = rest_pool_idx  # around other pooling target vertices
        self.pool_type = pool_type

    def forward(self, x):
        if self.pool_type in ["sum", "max", "mean"]:
            x = self.pool(x)
        else:
            x = self.downSample(x)
        return x

    def downSample(self, x):
        """
        Downsample without pooling
        """
        nv_prev = 10 * (4 ** self.level) + 2
        return x[:nv_prev, ...]

    def pool(self, x):
        """
        Pooling over either 6 or 7 vertices' features
        (+1 from self vertex)

        torch.Size([2562, 16, 64]) -> torch.Size([642, 16, 64])
        """
        # print('x.shape ', x.shape)

        n_vertices, n_batch, n_feats = x.shape  # torch.Size([2562, 16, 64])
        base_pool_idx = self.base_pool_idx.reshape(-1)
        out_base = x[base_pool_idx, ...]
        out_base = out_base.reshape((-1, 6, n_batch, n_feats))
        if self.pool_type == "sum":
            pooled_base = torch.sum(out_base, 1)
        elif self.pool_type == "mean":
            pooled_base = torch.mean(out_base, 1)
        elif self.pool_type == "max":
            pooled_base = torch.max(out_base, 1)[0]

        # rest_pool may be None (from L1 -> L0)
        if self.rest_pool_idx is not None:
            rest_pool_idx = self.rest_pool_idx.reshape(-1)
            out_rest = x[rest_pool_idx, ...]
            out_rest = out_rest.reshape((-1, 7, n_batch, n_feats))
            if self.pool_type == "sum":
                pooled_rest = torch.sum(out_rest, 1)
            elif self.pool_type == "mean":
                pooled_rest = torch.mean(out_rest, 1)
            elif self.pool_type == "max":
                pooled_rest = torch.max(out_rest, 1)[0]
            out = torch.cat((pooled_base, pooled_rest), dim=0)
        else:
            out = pooled_base

        # print('out.shape ', out.shape)

        return out

    def __repr__(self):
        return '{}(type={}, level={})'.format(self.__class__.__name__,
                                              self.pool_type,
                                              self.level)


class OutputBlock(nn.Module):
    """
    Input: [B x F]
    Output: [B x C]
    """

    def __init__(self, f_dim, h_dims, out_dim, dropout, debug=False):
        """ f_dim:     input feature dimension
            h_dims:    hidden dimensions
            out_dim:   output dimension
        """

        super(OutputBlock, self).__init__()
        self.debug = debug
        self.f_dim = f_dim
        self.hidden_dims = h_dims
        self.out_dim = out_dim
        self.dense_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.nonlinearity = F.relu
        curr_dim = f_dim
        next_dim = None
        # hidden_dims may be empty
        for h in self.hidden_dims:
            next_dim = h
            self.dense_layers.append(nn.Linear(curr_dim, next_dim))
            # if self.debug:
            #     self.batch_norms.append(nn.BatchNorm1d(next_dim, track_running_stats=False))
            # else:
            #     self.batch_norms.append(nn.BatchNorm1d(next_dim))
            curr_dim = next_dim

        # output layer
        next_dim = out_dim
        self.dense_layers.append(nn.Linear(curr_dim, next_dim))

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        """
        Input: [n_batch, n_vertex_L1, n_feats]
        """
        n_batch, n_vertex, n_feats = x.shape
        ind_layer_last = len(self.dense_layers) - 1
        for i, dense_layer in enumerate(self.dense_layers):
            if self.dropout is not None:
                x = self.dropout(x)
            x = dense_layer(x)
            # if i < ind_layer_last:
            #     x = self.batch_norms[i](x)
            #     x = self.nonlinearity(x)

        return x

    def __repr__(self):
        return '{}(layers={}\ndropout={}\n'.format(self.__class__.__name__,
                                                   self.dense_layers,
                                                   self.dropout)
