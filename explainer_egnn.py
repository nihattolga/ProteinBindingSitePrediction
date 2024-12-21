import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from models.models import EGNN  # Assuming you're using GAT
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import mpl_toolkits.mplot3d as plt3d
import random
import torch_geometric.transforms as T

# Constants
NODE_FEATURES_PATH = "biolip_node_features.npy"
NODE_LABELS_PATH = "biolip_node_labels.npy"
EDGE_INDEX_PATH = "biolip_edge_index.npy"
EDGE_FEATURES_PATH = "biolip_edge_features.npy"
PDB_ID_PATH = "biolip_pdb_id.npy"
CHAIN_ID_PATH = "biolip_chain_id.npy"
LIG_COORDS_PATH = "biolip_lig_coords.npy"
LIG_NAMES_PATH = "biolip_lig_names.npy"

MODEL_PATH = "weights/model_egnn_biolip.pth" 
BATCH_SIZE = 64

# Device configuration
device = torch.device("cpu")

def load_data():
    """Loads and preprocesses the data."""
    node_features = np.array(np.load(NODE_FEATURES_PATH, allow_pickle=True))
    node_labels = np.array(np.load(NODE_LABELS_PATH, allow_pickle=True))
    edge_index = np.array(np.load(EDGE_INDEX_PATH, allow_pickle=True))
    edge_features = np.array(np.load(EDGE_FEATURES_PATH, allow_pickle=True))
    pdb_ids = np.array(np.load(PDB_ID_PATH, allow_pickle=True))
    chain_ids = np.array(np.load(CHAIN_ID_PATH, allow_pickle=True))
    lig_coords = np.array(np.load(LIG_COORDS_PATH, allow_pickle=True))
    lig_names = np.array(np.load(LIG_NAMES_PATH, allow_pickle=True))

    # Create datalist
    datalist = []
    for i in range(len(node_features)):
        if len(
                edge_index[i]) != 0 and lig_coords[i].shape[0] != 0 and edge_features[i].shape[0] != 0:
            nf = torch.tensor(node_features[i][:, :-3], dtype=torch.float)
            nl = torch.tensor(node_labels[i], dtype=torch.float)
            ei = torch.tensor(edge_index[i], dtype=torch.long).T
            ef = torch.tensor(edge_features[i], dtype=torch.float)
            pdb_id = pdb_ids[i]
            chain_id = chain_ids[i]
            coords = torch.tensor(node_features[i][:, -3:], dtype=torch.float)
            l_coords = torch.tensor(lig_coords[i], dtype=torch.float)
            lig_name = lig_names[i]
            coords_label = closest_lig_coords(nl, l_coords, coords)
            datalist.append(
                Data(
                    x=nf,
                    edge_index=ei,
                    edge_attr=ef,
                    y=nl,
                    coords=coords,
                    coords_label=coords_label,
                    pdb_id=pdb_id,
                    chain_id=chain_id,
                    lig_coords=l_coords,
                    lig_name=lig_name))
    indices_with_empty_lig_coords = []

    for i, data in enumerate(datalist):
        if data.lig_coords.size(0) == 0:
            indices_with_empty_lig_coords.append(i)

    filtered_datalist = [data for i, data in enumerate(
        datalist) if i not in indices_with_empty_lig_coords]

    return filtered_datalist


def closest_lig_coords(labels, lig_coords, coords):
    """Finds the closest ligand coordinates for each binding site node."""
    binding_site_indices = torch.where((labels[:, 1] == 1) | (labels[:, 2] == 1))[0]
    binding_site_coords = coords[binding_site_indices]
    distances = torch.cdist(binding_site_coords, lig_coords)
    closest_lig_coords_indices = torch.argmin(distances, dim=1)
    closest_lig_coords = lig_coords[closest_lig_coords_indices]
    array = torch.zeros_like(coords)
    array[binding_site_indices] = closest_lig_coords
    return array


def initialize_model():
    """Initializes the GAT model."""
    model = EGNN(
        in_node_nf=30,
        hidden_nf=256,
        out_node_nf=3,
        in_edge_nf=3,
        device=device,
        attention=True).to(device)
    return model


def compute_node_feature_importance(
        model,
        data,
        num_samples=100,
        perturbation='zero'):
    """
    Compute node feature importance by permuting features and measuring the change in model output.

    :param model: The GAT model.
    :param data: The graph data (PyTorch Geometric Data object).
    :param num_samples: Number of samples to use for estimating importance.
    :param perturbation: Method of perturbation ('zero', 'permute', or 'noise').
    """
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')  # Get device
    model = model.to(device)  # Move model to the defined device.
    x, edge_index, edge_attr, coords = data.x.clone().to(device), data.edge_index.to(
        device), data.edge_attr.to(device), data.coords.to(device)
    original_output, _ = model(
        h=x, x=coords, edges=edge_index, edge_attr=edge_attr)
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
                x_permuted[:, feature_idx] = x[torch.randperm(
                    x.size(0)), feature_idx]
            elif perturbation == "noise":
                noise = torch.randn_like(x[:, feature_idx])
                x_permuted[:, feature_idx] += noise

            permuted_output, _ = model(
                h=x_permuted, x=coords, edges=edge_index, edge_attr=edge_attr)
            permuted_pred = torch.argmax(permuted_output, dim=1)
            importance = (original_pred != permuted_pred).float().mean().item()
            feature_importance[feature_idx] += importance / num_samples

    return feature_importance.cpu()


def plot_node_feature_importance(feature_importance, feature_names=None):
    """
    Plot node feature importance scores.

    :param feature_importance: Array of feature importance scores.
    :param feature_names: List of feature names corresponding to the scores.
    """
    if feature_names is None:
        feature_names = [
            f'Feature {i}' for i in range(
                len(feature_importance))]

    plt.barh(feature_names, feature_importance)
    plt.xlabel('Importance')
    plt.title('Feature Importance (EGNN)')
    plt.show()


def compute_edge_feature_importance(
        model,
        data,
        num_samples=100,
        perturbation='zero'):
    """
    Compute edge feature importance by perturbing edge features and measuring the change in model output.

    :param model: The GAT model.
    :param data: The graph data (PyTorch Geometric Data object).
    :param num_samples: Number of samples to use for estimating importance.
    :param perturbation: Method of perturbation ('zero', 'permute', or 'noise').
    :return: A tensor containing the importance scores for each edge feature.
    """
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')  # Get device
    model = model.to(device)  # Move model to the defined device.
    x, edge_index, edge_attr, coords = data.x.clone().to(device), data.edge_index.to(
        device), data.edge_attr.to(device), data.coords.to(device)
    original_output, _ = model(
        h=x, x=coords, edges=edge_index, edge_attr=edge_attr)
    original_pred = torch.argmax(original_output, dim=1)
    num_edge_features = edge_attr.size(1)

    importance_scores = torch.zeros(num_edge_features).to(device)
    if perturbation == 'zero':
        num_samples = 1
    for feature_idx in range(num_edge_features):
        edge_attr_permuted = edge_attr.clone()
        for _ in range(num_samples):
            if perturbation == 'zero':
                edge_attr_permuted[:, feature_idx] = 0
            elif perturbation == 'permute':
                edge_attr_permuted[:, feature_idx] = edge_attr[torch.randperm(
                    edge_attr.size(0)), feature_idx]
            elif perturbation == 'noise':
                noise = torch.randn_like(edge_attr[:, feature_idx])
                edge_attr_permuted[:, feature_idx] += noise

            permuted_output, _ = model(
                h=x, x=coords, edges=edge_index, edge_attr=edge_attr_permuted)
            permuted_pred = torch.argmax(permuted_output, dim=1)
            importance = (original_pred != permuted_pred).float().mean().item()
            importance_scores[feature_idx] += importance / num_samples

    return importance_scores.cpu()


def plot_edge_feature_importance(importance_scores, feature_names=None):
    """
    Plot edge feature importance scores.

    :param importance_scores: Tensor containing importance scores for each edge feature.
    :param feature_names: List of feature names corresponding to the scores.
    """
    if feature_names is None:
        feature_names = [
            f'Edge Feature {i}' for i in range(
                len(importance_scores))]

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importance_scores)
    plt.xlabel('Importance')
    plt.title('Edge Feature Importance (EGNN)')
    plt.show()


def compute_saliency_maps(model, data, feature_names=None):
    """
    Compute saliency maps for the input features.

    :param model: The GAT model.
    :param data: The graph data (PyTorch Geometric Data object).
    """
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')  # Get device
    model = model.to(device)  # Move model to the defined device.
    x, edge_index, edge_attr, coords = data.x.clone().to(device), data.edge_index.to(
        device), data.edge_attr.to(device), data.coords.to(device)
    model.eval()
    x.requires_grad_()
    output, _ = model(h=x, x=coords, edges=edge_index, edge_attr=edge_attr)
    loss = F.binary_cross_entropy_with_logits(output, output)
    loss.backward()

    saliency = x.grad.abs().detach().cpu().numpy()
    avg_saliency = np.mean(saliency, axis=0)  # Average across nodes

    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    plt.barh(feature_names, avg_saliency)  # Horizontal bar plot
    plt.xlabel('Average Saliency (EGNN)')
    plt.title('Average Saliency Map (Across Nodes)')
    plt.tight_layout()
    plt.show()


def compute_coord_importance(
        model,
        data,
        num_samples=100,
        perturbation='zero'):
    """
    Compute 3D coordinate importance by perturbing coordinates and measuring changes in model output.

    :param model: The EGNN model.
    :param data: The graph data (PyTorch Geometric Data object).
    :param num_samples: Number of samples to use for estimating importance.
    :param perturbation: Method of perturbation ('zero', 'permute', or 'noise').
    :return: A tensor containing the importance scores for each coordinate dimension (x, y, z).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x, edge_index, edge_attr, coords = data.x.clone().to(device), data.edge_index.to(
        device), data.edge_attr.to(device), data.coords.clone().to(device)
    original_output, _ = model(
        h=x, x=coords, edges=edge_index, edge_attr=edge_attr)  # Get original output
    original_pred = torch.argmax(original_output, dim=1)

    coord_importance = torch.zeros(coords.size(1)).to(
        device)  # Initialize importance for x, y, z
    if perturbation == 'zero':
        num_samples = 1

    for coord_idx in range(
            coords.size(1)):  # Iterate through x, y, z dimensions
        coords_permuted = coords.clone()
        for _ in range(num_samples):
            if perturbation == "zero":
                coords_permuted[:, coord_idx] = 0
            elif perturbation == "permute":
                coords_permuted[:, coord_idx] = coords[torch.randperm(
                    coords.size(0)), coord_idx]
            elif perturbation == "noise":
                # Scale noise appropriately
                noise = torch.randn_like(coords[:, coord_idx]) * 0.1
                coords_permuted[:, coord_idx] += noise

            permuted_output, _ = model(
                h=x, x=coords_permuted, edges=edge_index, edge_attr=edge_attr)
            permuted_pred = torch.argmax(permuted_output, dim=1)
            importance = (original_pred != permuted_pred).float().mean().item()
            coord_importance[coord_idx] += importance / num_samples

    return coord_importance.cpu()


def plot_coord_importance(coord_importance):
    """Plot 3D coordinate importance scores."""
    coord_names = ['x', 'y', 'z']
    plt.barh(coord_names, coord_importance)
    plt.xlabel('Importance')
    plt.title('3D Coordinate Importance (EGNN)')
    plt.show()


if __name__ == "__main__":
    datalist = load_data()
    model = initialize_model()
    model.load_state_dict(torch.load(MODEL_PATH))
    data_size = len(datalist)
    train_size = int(0.8 * data_size)
    val_size = int(0.1 * data_size)
    data = Batch.from_data_list(datalist[train_size + val_size:])

    feature_importance, edge_features_importance = [], []
    for idx, data in enumerate(datalist[train_size + val_size:]):
        data = data.to(device)

        edge_features_importance.append(compute_edge_feature_importance(model, data, num_samples=100, perturbation='permute'))
        print(data.pdb_id)
    feature_importance = np.mean(feature_importance, axis=0)
    edge_features_importance = np.mean(edge_features_importance, axis=0)
    edge_features_names = ["distance", "cosine_angle", "sequence"]
    feature_names = [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLU",
        "GLN",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "UNK",
        "phi",
        "psi",
        "omega",
        "tau",
        "theta",
        "atom_sasa",
        "alpha-helix",
        "beta-sheet",
        "coil"]
    plot_edge_feature_importance(edge_features_importance, edge_features_names)
    compute_saliency_maps(model, data, feature_names)

    coord_importance_list = []
    for idx, data in enumerate(
            datalist[train_size + val_size:]): 
        data = data.to(device)
        coord_importance = compute_coord_importance(
            model, data, num_samples=25, perturbation='permute')
        coord_importance_list.append(coord_importance)
        print(idx)

    avg_coord_importance = torch.stack(coord_importance_list).mean(
        dim=0)  
    plot_coord_importance(avg_coord_importance)
