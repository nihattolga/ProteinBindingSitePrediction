import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from models.models import GCN
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import mpl_toolkits.mplot3d as plt3d
import random
import torch_geometric.transforms as T

NODE_FEATURES_PATH = "biolip_node_features.npy"
NODE_LABELS_PATH = "biolip_node_labels.npy"
EDGE_INDEX_PATH = "biolip_edge_index.npy"
EDGE_FEATURES_PATH = "biolip_edge_features.npy"
MODEL_PATH = "weights/model_gcn_biolip.pth"
BATCH_SIZE = 64

device = torch.device("cpu")


def load_data():
    node_features = np.load(NODE_FEATURES_PATH, allow_pickle=True)
    node_labels = np.load(NODE_LABELS_PATH, allow_pickle=True)
    edge_index = np.load(EDGE_INDEX_PATH, allow_pickle=True)
    edge_features = np.load(EDGE_FEATURES_PATH, allow_pickle=True)

    datalist = []
    for i in range(len(node_features)):
        if len(edge_index[i]) != 0 and edge_features[i].shape[0] != 0:
            nf = torch.tensor(node_features[i], dtype=torch.float)
            nl = torch.tensor(node_labels[i], dtype=torch.float)
            ei = torch.tensor(edge_index[i], dtype=torch.long).T
            ef = torch.tensor(edge_features[i], dtype=torch.float)
            pre_transform = T.Compose([T.GCNNorm(), T.ToSparseTensor()])
            d = Data(x=nf, edge_index=ei, edge_attr=ef, y=nl)
            d = pre_transform(d)
            datalist.append(d)

    return datalist


def initialize_model():
    model = GCN(
        hidden_channels=256,
        num_layers=2,
        alpha=0.5,
        theta=1.0,
        shared_weights=False,
        dropout=0.2
    ).to(device)
    return model


def visualize_attention(model, data, layer_idx=0, node_idx=None, head_idx=0):
    x, edge_index, edge_attr, y = data.x, data.edge_index, data.edge_attr, data.y
    model.eval()
    with torch.no_grad():
        for i in range(layer_idx + 1):
            x, alpha = model.convs[i](
                x, edge_index, edge_attr, return_attention_weights=True
            )

    alpha = alpha[head_idx].numpy()

    if node_idx is not None:
        attention = alpha[edge_index[1] == node_idx]
        plt.hist(attention, bins=20)
        plt.title(f"Attention Scores for Node {node_idx}")
        plt.xlabel("Attention Score")
        plt.ylabel("Frequency")
        plt.show()
    else:
        G = nx.Graph()
        edges = edge_index.T.tolist()
        G.add_edges_from(edges)
        weights = alpha / alpha.max()
        edge_colors = plt.cm.Blues(weights.flatten())

        node_colors = [
            'red' if np.argmax(label) == 0
            else 'green' if np.argmax(label) == 1
            else 'blue'
            for label in y
        ]

        pos = nx.spring_layout(G)
        nx.draw(
            G, pos, with_labels=True, width=weights.flatten() * 10,
            edge_color=edge_colors, edge_cmap=plt.cm.Greys, node_size=50,
            node_color=node_colors, font_size=4
        )
        plt.title(f"Attention Weights (Layer {layer_idx}, Head {head_idx})")
        plt.show()


def compute_node_feature_importance(model, data, num_samples=100, perturbation='zero'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x, adj_t = data.x.clone().to(device), data.adj_t.to(device)
    original_output = model(x, adj_t).detach()
    original_pred = torch.argmax(original_output, dim=1)

    feature_importance = torch.zeros(x.size(1)).to(device)
    if perturbation == 'zero':
        num_samples = 1
    for feature_idx in range(x.size(1)):
        x_permuted = x.clone()
        for _ in range(num_samples):
            if perturbation == "zero":
                x_permuted[:, feature_idx] = 0
            elif perturbation == "permute":
                x_permuted[:, feature_idx] = x[torch.randperm(x.size(0)), feature_idx]
            elif perturbation == "noise":
                noise = torch.randn_like(x[:, feature_idx])
                x_permuted[:, feature_idx] += noise

            permuted_output = model(x_permuted, adj_t).detach()
            permuted_pred = torch.argmax(permuted_output, dim=1)
            importance = (original_pred != permuted_pred).float().mean().item()
            feature_importance[feature_idx] += importance / num_samples

    return feature_importance.cpu()


def plot_node_feature_importance(feature_importance, feature_names=None):
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(feature_importance))]

    plt.barh(feature_names, feature_importance)
    plt.xlabel('Importance')
    plt.title('Feature Importance (GCN)')
    plt.show()


if __name__ == "__main__":
    datalist = load_data()
    model = initialize_model()
    model.load_state_dict(torch.load(MODEL_PATH))
    data_size = len(datalist)
    train_size = int(0.8 * data_size)
    val_size = int(0.1 * data_size)
    data = Batch.from_data_list(datalist[train_size + val_size:])

    data = data.to(device)
    feature_names = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "UNK", "phi", "psi", "omega", "tau", "theta", "atom_sasa", "alpha-helix",
        "beta-sheet", "coil", "x", "y", "z"
    ]
    feature_importance = compute_node_feature_importance(
        model, data, num_samples=25, perturbation='permute'
    )
    plot_node_feature_importance(feature_importance, feature_names)
