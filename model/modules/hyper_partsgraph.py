import math

import torch
from torch import nn
import numpy as np
from einops import rearrange

CONNECTIONS = {10: [9], 9: [8, 10], 8: [7, 9], 14: [15, 8], 15: [16, 14], 11: [12, 8], 12: [13, 11],
               7: [0, 8], 0: [1, 7], 1: [2, 0], 2: [3, 1], 4: [5, 0], 5: [6, 4], 16: [15], 13: [12], 3: [2], 6: [5]}


class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, num_nodes, neighbour_num=4, mode='spatial', use_temporal_similarity=True,
                 temporal_connection_len=1, connections=None):
        self.nodes_ = """
        :param dim_int: Channel input dimension
        :param dim_out: Channel output dimension
        :param num_nodes: Number of nodes
        :param neighbour_num: Neighbor numbers. Used in temporal GCN to create edges
        :param mode: Either 'spatial' or 'temporal'
        :param use_temporal_similarity: If true, for temporal GCN uses top-k similarity between nodes
        :param temporal_connection_len: Connects joint to itself within next `temporal_connection_len` frames
        :param connections: Spatial connections for graph edges (Optional)
        """
        super().__init__()
        assert mode in ['spatial', 'temporal'], "Mode is undefined"

        self.relu = nn.ReLU()
        self.neighbour_num = neighbour_num
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.mode = mode
        self.use_temporal_similarity = use_temporal_similarity
        self.num_nodes = num_nodes
        self.connections = connections

        self.U = nn.Linear(self.dim_in, self.dim_out)
        self.V = nn.Linear(self.dim_in, self.dim_out)
        self.batch_norm = nn.BatchNorm1d(self.num_nodes)

        self._init_gcn()

        if mode == 'spatial':
            self.adj = self._init_spatial_adj()
        elif mode == 'temporal' and not self.use_temporal_similarity:
            self.adj = self._init_temporal_adj(temporal_connection_len)

    def _init_gcn(self):
        self.U.weight.data.normal_(0, math.sqrt(2. / self.dim_in))
        self.V.weight.data.normal_(0, math.sqrt(2. / self.dim_in))
        self.batch_norm.weight.data.fill_(1)
        self.batch_norm.bias.data.zero_()

    def _init_spatial_adj(self):
        adj = torch.zeros((self.num_nodes, self.num_nodes))
        connections = self.connections if self.connections is not None else CONNECTIONS

        for i in range(self.num_nodes):
            connected_nodes = connections[i]
            for j in connected_nodes:
                adj[i, j] = 1
        return adj

    def _init_temporal_adj(self, connection_length):
        """Connects each joint to itself and the same joint withing next `connection_length` frames."""
        adj = torch.zeros((self.num_nodes, self.num_nodes))

        for i in range(self.num_nodes):
            try:
                for j in range(connection_length + 1):
                    adj[i, i + j] = 1
            except IndexError:  # next j frame does not exist
                pass
        return adj

    @staticmethod
    def normalize_digraph(adj):
        b, n, c = adj.shape

        node_degrees = adj.detach().sum(dim=-1)
        deg_inv_sqrt = node_degrees ** -0.5
        norm_deg_matrix = torch.eye(n)
        dev = adj.get_device()
        if dev >= 0:
            norm_deg_matrix = norm_deg_matrix.to(dev)
        norm_deg_matrix = norm_deg_matrix.view(1, n, n) * deg_inv_sqrt.view(b, n, 1)
        norm_adj = torch.bmm(torch.bmm(norm_deg_matrix, adj), norm_deg_matrix)

        return norm_adj

    def change_adj_device_to_cuda(self, adj):
        dev = self.V.weight.get_device()
        if dev >= 0 and adj.get_device() < 0:
            adj = adj.to(dev)
        return adj

    def forward(self, x):
        """
        x: tensor with shape [B, T, J, C]
        """
        b, t, j, c = x.shape
        if self.mode == 'temporal':
            x = x.transpose(1, 2)  # (B, T, J, C) -> (B, J, T, C)
            x = x.reshape(-1, t, c)
            if self.use_temporal_similarity:
                similarity = x @ x.transpose(1, 2)
                threshold = similarity.topk(k=self.neighbour_num, dim=-1, largest=True)[0][..., -1].view(b * j, t, 1)
                adj = (similarity >= threshold).float()
            else:
                adj = self.adj
                adj = self.change_adj_device_to_cuda(adj)
                adj = adj.repeat(b * j, 1, 1)

        else:
            x = x.reshape(-1, j, c)  # bt j c
            adj = self.adj
            adj = self.change_adj_device_to_cuda(adj)
            adj = adj.repeat(b * t, 1, 1)  # bt j j

        norm_adj = self.normalize_digraph(adj)
        aggregate = norm_adj @ self.V(x)

        if self.dim_in == self.dim_out:
            x = self.relu(x + self.batch_norm(aggregate + self.U(x)))
        else:
            x = self.relu(self.batch_norm(aggregate + self.U(x)))

        x = x.reshape(-1, t, j, self.dim_out) if self.mode == 'spatial' \
            else x.reshape(-1, j, t, self.dim_out).transpose(1, 2)
        return x



class HyperGCN(nn.Module):
    def __init__(self, dim_in, dim_out, num_nodes, neighbour_num=4, mode='spatial',
                 use_partscale=True,
                 use_bodyscale=False,
                 use_temporal_similarity=True,
                 temporal_connection_len=1, connections=None, dataset='h36m'):
        self.nodes_ = """
                :param dim_int: Channel input dimension
                :param dim_out: Channel output dimension
                :param num_nodes: Number of nodes
                :param neighbour_num: Neighbor numbers. Used in temporal GCN to create edges
                :param mode: Either 'spatial' or 'temporal' or ''
                :param use_partscale: sue body hyper graph conv under spatial mode
                :param use_bodyscale: use body hyper graph conv under spatial mode
                :param use_temporal_similarity: If true, for temporal GCN uses top-k similarity between nodes, under temporal mode
                :param temporal_connection_len: Connects joint to itself within next `temporal_connection_len` frames
                :param connections: Spatial connections for graph edges (Optional)
                :dataset : h36m, mpi
                """
        super().__init__()
        assert mode in ['spatial', 'temporal'], "Mode is undefined"

        self.relu = nn.ReLU()
        self.neighbour_num = neighbour_num
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.mode = mode
        self.use_partscale = use_partscale
        self.use_bodyscale = use_bodyscale
        self.use_temporal_similarity = use_temporal_similarity
        self.num_nodes = num_nodes
        self.connections = connections
        self.dataset = dataset

        self.U = nn.Linear(self.dim_in, self.dim_out)
        self.V = nn.Linear(self.dim_in, self.dim_out)

        self.batch_norm = nn.BatchNorm1d(self.num_nodes)

        self._init_gcn()

        if mode == 'spatial':
            self.adj = self._init_spatial_adj()
            if use_partscale:
                G_part = self._init_part_adj()
                self.p_adj = nn.Parameter(torch.from_numpy(G_part.astype(np.float32)))
                self.conv_part = nn.Conv2d(self.dim_in, self.dim_out, 1)
            if use_bodyscale:
                G_body = self._init_body_adj()
                self.b_adj = nn.Parameter(torch.from_numpy(G_body.astype(np.float32)))

        elif mode == 'temporal' and not self.use_temporal_similarity:
            self.adj = self._init_temporal_adj(temporal_connection_len)


    def _init_gcn(self):
        self.U.weight.data.normal_(0, math.sqrt(2. / self.dim_in))
        self.V.weight.data.normal_(0, math.sqrt(2. / self.dim_in))
        self.batch_norm.weight.data.fill_(1)
        self.batch_norm.bias.data.zero_()

    def _init_spatial_adj(self):
        adj = torch.zeros((self.num_nodes, self.num_nodes))
        connections = self.connections if self.connections is not None else CONNECTIONS

        for i in range(self.num_nodes):
            connected_nodes = connections[i]
            for j in connected_nodes:
                adj[i, j] = 1
        return adj   # j j   not normalizized

    def _init_part_adj(self, variable_weight=False):
        H = np.zeros((self.num_nodes, 10))
        if self.dataset == 'h36m':
            # part 1:
            H[0][0],H[7][0],H[8][0] = 1,1,1
            # part 2:
            H[8][1], H[9][1], H[10][1] = 1,1,1
            # part 3:
            H[0][2], H[1][2], H[2][2] = 1,1,1
            # part 4:
            H[2][3], H[3][3] = 1,1
            # part 5:
            H[0][4],H[4][4],H[5][4] = 1,1,1
            # part 6:
            H[5][5], H[6][5] = 1,1
            # part 7:
            H[8][6], H[14][6], H[15][6] = 1,1,1
            # part 8:
            H[8][7],H[11][7],H[12][7] = 1,1,1
            # part 9:
            H[15][8],H[16][8] = 1,1
            # part 10:
            H[12][9], H[13][9] = 1,1
        elif self.dataset == 'mpi':
            # part 1:
            H[1][0], H[14][0], H[15][0] = 1, 1, 1
            # part 2:
            H[0][1], H[1][1], H[16][1] = 1, 1, 1
            # part 3:
            H[14][2], H[11][2], H[12][2] = 1, 1, 1
            # part 4:
            H[12][3], H[13][3] = 1, 1
            # part 5:
            H[14][4], H[8][4], H[9][4] = 1, 1, 1
            # part 6:
            H[9][5], H[10][5] = 1, 1
            # part 7:
            H[1][6], H[5][6], H[6][6] = 1, 1, 1
            # part 8:
            H[1][7], H[2][7], H[3][7] = 1, 1, 1
            # part 9:
            H[6][8], H[7][8] = 1, 1
            # part 10:
            H[3][9], H[4][9] = 1, 1

        H = np.array(H)
        n_edge = H.shape[1]
        # the weight of the hyperedge
        W = np.ones(n_edge)
        # the degree of the node
        DV = np.sum(H * W, axis=1)
        # the degree of the hyperedge
        DE = np.sum(H, axis=0)

        invDE = np.mat(np.diag(np.power(DE, -1)))
        DV2 = np.mat(np.diag(np.power(DV, -0.5)))
        W = np.mat(np.diag(W))
        H = np.mat(H)
        HT = H.T

        if variable_weight:
            DV2_H = DV2 * H
            invDE_HT_DV2 = invDE * HT * DV2
            return DV2_H, W, invDE_HT_DV2
        else:
            G = DV2 * H * W * invDE * HT * DV2
            return G

    def _init_body_adj(self):
        return 0

    def _init_temporal_adj(self, connection_length):
        """Connects each joint to itself and the same joint withing next `connection_length` frames."""
        adj = torch.zeros((self.num_nodes, self.num_nodes))

        for i in range(self.num_nodes):
            try:
                for j in range(connection_length + 1):
                    adj[i, i + j] = 1
            except IndexError:  # next j frame does not exist
                pass
        return adj

    @staticmethod
    def normalize_digraph(adj):
        b, n, c = adj.shape  # batchsize, j, j

        node_degrees = adj.detach().sum(dim=-1)
        deg_inv_sqrt = node_degrees ** -0.5
        norm_deg_matrix = torch.eye(n)
        dev = adj.get_device()
        if dev >= 0:
            norm_deg_matrix = norm_deg_matrix.to(dev)
        norm_deg_matrix = norm_deg_matrix.view(1, n, n) * deg_inv_sqrt.view(b, n, 1)
        norm_adj = torch.bmm(torch.bmm(norm_deg_matrix, adj), norm_deg_matrix)

        return norm_adj

    def change_adj_device_to_cuda(self, adj):
        dev = self.V.weight.get_device()
        if dev >= 0 and adj.get_device() < 0:
            adj = adj.to(dev)
        return adj

    def norm(self, A):
        D_list = torch.sum(A, 0).view(1, self.num_nodes)
        D_list_12 = (D_list + 0.001)**(-1)
        D_12 = torch.eye(self.num_nodes).to(device=A.device) * D_list_12
        A = torch.matmul(A, D_12)
        return A

    def forward(self, x):
        """
        x: tensor with shape [B, T, J, C]
        """
        b, t, j, c = x.shape
        if self.mode == 'temporal':
            x = x.transpose(1, 2)  # (B, T, J, C) -> (B, J, T, C)
            x = x.reshape(-1, t, c)
            if self.use_temporal_similarity:
                similarity = x @ x.transpose(1, 2)
                threshold = similarity.topk(k=self.neighbour_num, dim=-1, largest=True)[0][..., -1].view(b * j, t, 1)
                adj = (similarity >= threshold).float()
            else:
                adj = self.adj
                adj = self.change_adj_device_to_cuda(adj)
                adj = adj.repeat(b * j, 1, 1)
            norm_adj = self.normalize_digraph(adj)
            aggregate = norm_adj @ self.V(x)

        else:
            if self.use_partscale:
                G_part = self.p_adj.cuda(x.get_device())   # v v , x shape : b t v c
                x_part = rearrange(x, 'b t v c -> b (c t) v')
                x_part = torch.matmul(x_part, self.norm(G_part))
                x_part = rearrange(x_part, 'b (c t) v -> b c t v',  c=c)
                aggregate = self.conv_part(x_part)  # b cout t v
                aggregate = rearrange(aggregate, 'b c t v -> (b t) v c')
                x = rearrange(x, 'b t v c -> (b t) v c')
            elif self.use_bodyscale:
                pass
            else:
                x = x.reshape(-1, j, c)
                adj = self.adj
                adj = self.change_adj_device_to_cuda(adj)
                adj = adj.repeat(b * t, 1, 1)

                norm_adj = self.normalize_digraph(adj)
                aggregate = norm_adj @ self.V(x)

        if self.dim_in == self.dim_out:
            x = self.relu(x + self.batch_norm(aggregate + self.U(x)))
        else:
            x = self.relu(self.batch_norm(aggregate + self.U(x)))

        x = x.reshape(-1, t, j, self.dim_out) if self.mode == 'spatial' \
            else x.reshape(-1, j, t, self.dim_out).transpose(1, 2)
        return x


# test

if __name__ == '__main__':
    net = HyperGCN(dim_in=64, dim_out=64, num_nodes=17, neighbour_num=4, mode='spatial',
                 use_partscale=True,
                 use_bodyscale=False,
                 use_temporal_similarity=True,
                 temporal_connection_len=1, connections=None, dataset='h36m').cuda()
    input = torch.randn(2, 81, 17, 64).cuda()

    print(net(input).shape)
