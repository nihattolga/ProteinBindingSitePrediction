import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCN2Conv

"""
EGNN taken from the paper https://arxiv.org/abs/2102.09844
"""
class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer.
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        act_fn=nn.SiLU(),
        residual=True,
        attention=False,
        normalize=False,
        coords_agg="mean",
        tanh=False,
    ):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, nn.Linear(hidden_nf, 1, bias=False)]
        if tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid(),
            )

    def edge_model(self, source, target, radial, edge_attr):
        inputs = [source, target, radial]
        if edge_attr is not None:
            inputs.append(edge_attr)
        out = self.edge_mlp(torch.cat(inputs, dim=1))
        if self.attention:
            out *= self.att_mlp(out)
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, _ = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        inputs = [x, agg]
        if node_attr is not None:
            inputs.append(node_attr)
        out = self.node_mlp(torch.cat(inputs, dim=1))
        if self.residual:
            out += x
        return out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, _ = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        agg_func = unsorted_segment_sum if self.coords_agg == "sum" else unsorted_segment_mean
        coord += agg_func(trans, row, num_segments=coord.size(0))
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = (coord_diff**2).sum(dim=1, keepdim=True)
        if self.normalize:
            norm = radial.sqrt().detach() + self.epsilon
            coord_diff /= norm
        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord, edge_attr


class EGNN(nn.Module):
    """
    E(n) Equivariant Graph Neural Network.
    """

    def __init__(
        self,
        in_node_nf,
        hidden_nf,
        out_node_nf,
        in_edge_nf=0,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=4,
        residual=True,
        attention=False,
        normalize=False,
        tanh=False,
    ):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, hidden_nf)
        self.embedding_out = nn.Linear(hidden_nf, out_node_nf)

        for i in range(n_layers):
            layer = E_GCL(
                hidden_nf,
                hidden_nf,
                hidden_nf,
                edges_in_d=in_edge_nf,
                act_fn=act_fn,
                residual=residual,
                attention=attention,
                normalize=normalize,
                tanh=tanh,
            )
            self.add_module(f"gcl_{i}", layer)

        self.to(device)

    def forward(self, h, x, edges, edge_attr):
        h = self.embedding_in(h)
        for i in range(self.n_layers):
            h, x, _ = self._modules[f"gcl_{i}"](h, edges, x, edge_attr=edge_attr)
        h = self.embedding_out(h)
        return h, x


def unsorted_segment_sum(data, segment_ids, num_segments):
    result = data.new_zeros((num_segments, data.size(1)))
    segment_ids = segment_ids.unsqueeze(-1).expand_as(data)
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result = data.new_zeros((num_segments, data.size(1)))
    count = data.new_zeros((num_segments, data.size(1)))
    segment_ids = segment_ids.unsqueeze(-1).expand_as(data)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1e-6)


class GAT(nn.Module):
    """
    Graph Attention Network.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.0, num_layers=2):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.lins.append(nn.Linear(in_channels, hidden_channels * heads))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
            self.lins.append(nn.Linear(hidden_channels * heads, hidden_channels * heads))

        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=6, dropout=0.0, concat=False))
        self.lins.append(nn.Linear(hidden_channels * heads, out_channels))

    def forward(self, x, edge_index, edge_attr):
        for i in range(self.num_layers - 1):
            x = F.relu(self.convs[i](x, edge_index, edge_attr) + self.lins[i](x))
        x = self.convs[-1](x, edge_index, edge_attr) + self.lins[-1](x)
        return x


class GCN(nn.Module):
    """
    Graph Convolutional Network.
    """

    def __init__(self, hidden_channels, num_layers, alpha, theta, shared_weights=True, dropout=0.0):
        super(GCN, self).__init__()
        self.lins = nn.ModuleList()
        self.convs = nn.ModuleList()

        self.lins.append(nn.Linear(33, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, 3))

        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(
                    hidden_channels,
                    alpha,
                    theta,
                    layer + 1,
                    shared_weights,
                    normalize=False,
                )
            )

        self.dropout = dropout

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x_0 = x = self.lins[0](x).relu()

        for conv in self.convs:
            h = F.dropout(x, self.dropout, training=self.training)
            x = x + conv(h, x_0, adj_t).relu()

        x = F.dropout(x, self.dropout, training=self.training)
        return self.lins[1](x)
