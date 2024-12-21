import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from models.models import GAT  # Assuming you're using GAT
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import mpl_toolkits.mplot3d as plt3d
import random

# Constants
NODE_FEATURES_PATH = "biolip_node_features.npy"
NODE_LABELS_PATH = "biolip_node_labels.npy"
EDGE_INDEX_PATH = "biolip_edge_index.npy"
EDGE_FEATURES_PATH = "biolip_edge_features.npy"

MODEL_PATH = "weights/model_gat_ss_biolip.pth"  # Path to your trained GAT model
BATCH_SIZE = 64

# Device configuration
device = torch.device("cpu")

# --- Data Loading and Preprocessing ---
def load_data():
    """Loads and preprocesses the data."""
    node_features = np.array(np.load(NODE_FEATURES_PATH, allow_pickle=True))
    node_labels = np.array(np.load(NODE_LABELS_PATH, allow_pickle=True))
    edge_index = np.array(np.load(EDGE_INDEX_PATH, allow_pickle=True))
    edge_features = np.array(np.load(EDGE_FEATURES_PATH, allow_pickle=True))

    # Create datalist
    datalist = []
    for i in range(len(node_features)):
        if len(edge_index[i]) != 0 and edge_features[i].shape[0] != 0:
            nf = torch.tensor(node_features[i], dtype=torch.float)
            nl = torch.tensor(node_labels[i], dtype=torch.float)
            ei = torch.tensor(edge_index[i], dtype=torch.long).T
            ef = torch.tensor(edge_features[i], dtype=torch.float)
            datalist.append(Data(x=nf, edge_index=ei, edge_attr=ef, y=nl))
    
    return datalist

# --- Model Initialization ---
def initialize_model():
    """Initializes the GAT model."""
    model = GAT(num_layers=2, in_channels=33, hidden_channels=256, out_channels=3, dropout=0.2).to(device)
    return model

def visualize_attention(model, data, layer_idx=0, node_idx=None, head_idx=0):
    """
    Visualize attention coefficients for a specific node or all nodes.

    :param model: The GAT model.
    :param data: The graph data (PyTorch Geometric Data object).
    :param layer_idx: The index of the GAT layer to visualize.
    :param node_idx: The index of the node to visualize. If None, visualize all nodes.
    :param head_idx: The index of the attention head to visualize.
    """
    x, edge_index, edge_attr, y = data.x, data.edge_index, data.edge_attr, data.y
    model.eval()
    with torch.no_grad():
        for i in range(layer_idx + 1):
            # Forward pass to get attention coefficients
            x, alpha = model.convs[i](x, edge_index, edge_attr, return_attention_weights=True)

    alpha = alpha[head_idx].numpy() 

    if node_idx is not None:
        attention = alpha[edge_index[1] == node_idx]
        # Plotting the attention scores for the selected node
        plt.hist(attention, bins=20)
        plt.title(f"Attention Scores for Node {node_idx} (Layer {layer_idx}, Head {head_idx})")
        plt.xlabel("Attention Score")
        plt.ylabel("Frequency")
        plt.show()
    else:
        # Visualize all nodes' attention scores
        G = nx.Graph()
        # Reshape edge_index to get pairs of nodes
        edges = edge_index.T.tolist()  # Transpose and convert to list of lists
        G.add_edges_from(edges)  # Add edges to the graph
        weights = alpha / alpha.max()  # Normalize weights
        # Map weights to a colormap
        edge_colors = plt.cm.Blues(weights.flatten())
        
        # Determine node colors based on data.y
        node_colors = ['red' if np.argmax(label) == 0 
                    else 'green' if np.argmax(label) == 1 
                    else 'blue' for label in y]
        
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, width=weights.flatten() * 10, 
                edge_color=edge_colors, edge_cmap=plt.cm.Greys, node_size=50, 
                node_color=node_colors, font_size=4) # Add node_color argument
        plt.title(f"Attention Weights (Layer {layer_idx}, Head {head_idx})")
        plt.show()

def visualize_attention_3d(model, data, layer_idx=0, head_idx=0):
    x, edge_index, edge_attr, y = data.x, data.edge_index, data.edge_attr, data.y
    model.eval()
    # Extract 3D coordinates from data.x
    coordinates = x[:, -3:].detach().numpy()
    with torch.no_grad():
        for i in range(layer_idx + 1):
            # Forward pass to get attention coefficients
            x, alpha = model.convs[i](x, edge_index, edge_attr, return_attention_weights=True)
    #Calculate attention weights
    attention_weights = alpha[1].cpu().numpy()

    # Determine node colors based on data.y
    node_colors = ['orange' if np.argmax(label) == 0 
                else 'green' if np.argmax(label) == 1 
                else 'red' for label in y]
    
    # Create 3D graph
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Set background color to black
    ax.xaxis.set_pane_color((0.2, 0.2, 0.2, 0.2))
    ax.yaxis.set_pane_color((0.2, 0.2, 0.2, 0.2))
    ax.zaxis.set_pane_color((0.2, 0.2, 0.2, 0.2))
    # Normalize attention weights
    weights = attention_weights / attention_weights.max()
    # Reshape edge_index to get pairs of nodes
    edges = edge_index.T.tolist() 

    # Plot edges
    for i, edge in enumerate(edges):
        source_node = edge[0]
        target_node = edge[1]
        x_coords = [coordinates[source_node][0], coordinates[target_node][0]]
        y_coords = [coordinates[source_node][1], coordinates[target_node][1]]
        z_coords = [coordinates[source_node][2], coordinates[target_node][2]]
        ax.plot(x_coords, y_coords, z_coords, c=plt.cm.Blues(weights[i][head_idx]*2), linewidth=weights[i][head_idx]*2)

    # Plot nodes
    ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], 
            c=node_colors, s=10) 
    # Adjust view for zooming
    ax.set_box_aspect([1,1,1]) 
    ax.dist = 7 

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f"Attention Weights (Layer {layer_idx}, Head {head_idx}) - 3D View")
    plt.show()

def compute_node_feature_importance(model, data, num_samples=100, perturbation='zero'):
    """
    Compute node feature importance by permuting features and measuring the change in model output.

    :param model: The GAT model.
    :param data: The graph data (PyTorch Geometric Data object).
    :param num_samples: Number of samples to use for estimating importance.
    :param perturbation: Method of perturbation ('zero', 'permute', or 'noise').
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model = model.to(device) 
    x, edge_index, edge_attr = data.x.clone().to(device), data.edge_index.to(device), data.edge_attr.to(device)
    original_output = model(x, edge_index, edge_attr).detach()
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
            
            permuted_output = model(x_permuted, edge_index, edge_attr).detach()
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
        feature_names = [f'Feature {i}' for i in range(len(feature_importance))]
    
    plt.barh(feature_names, feature_importance)
    plt.xlabel('Importance')
    plt.title('Feature Importance (GAT - Semi supervised)')
    plt.show()

def compute_edge_feature_importance(model, data, num_samples=100, perturbation='zero'):
    """
    Compute edge feature importance by perturbing edge features and measuring the change in model output.

    :param model: The GAT model.
    :param data: The graph data (PyTorch Geometric Data object).
    :param num_samples: Number of samples to use for estimating importance.
    :param perturbation: Method of perturbation ('zero', 'permute', or 'noise').
    :return: A tensor containing the importance scores for each edge feature.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x, edge_index, edge_attr = data.x.clone().to(device), data.edge_index.to(device), data.edge_attr.to(device)
    original_output = model(x, edge_index, edge_attr).detach()
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
                edge_attr_permuted[:, feature_idx] = edge_attr[torch.randperm(edge_attr.size(0)), feature_idx]
            elif perturbation == 'noise':
                noise = torch.randn_like(edge_attr[:, feature_idx])
                edge_attr_permuted[:, feature_idx] += noise

            permuted_output = model(x, edge_index, edge_attr_permuted).detach()
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
        feature_names = [f'Edge Feature {i}' for i in range(len(importance_scores))]
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importance_scores.cpu().numpy())
    plt.xlabel('Importance')
    plt.title('Edge Feature Importance (GAT - Semi supervised)')
    plt.show()

def compute_saliency_maps(model, data, feature_names=None):
    """
    Compute saliency maps for the input features.

    :param model: The GAT model.
    :param data: The graph data (PyTorch Geometric Data object).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    model.eval()
    x.requires_grad_()
    output = model(x, edge_index, edge_attr)
    loss = F.binary_cross_entropy_with_logits(output, output)
    loss.backward()

    saliency = x.grad.abs().detach().cpu().numpy()
    avg_saliency = np.mean(saliency, axis=0) # Average across nodes

    plt.figure(figsize=(10, 6))  
    plt.barh(feature_names, avg_saliency) 
    plt.xlabel('Average Saliency (GAT - Semi supervised)')
    plt.title('Average Saliency Map (Across Nodes)')
    plt.tight_layout()
    plt.show()


def average_attention_weights(model, data, layer_idx=0, head_idx=0):
    x, edge_index, edge_attr, y = data.x, data.edge_index, data.edge_attr, data.y
    model.eval()
    # Forward pass to get attention coefficients
    with torch.no_grad():
        for i in range(layer_idx + 1):
            x, alpha = model.convs[i](x, edge_index, edge_attr, return_attention_weights=True)

    # Calculate attention weights
    attention_weights = alpha[1].cpu().numpy()

    # Convert one-hot encoded labels to class indices
    class_labels = torch.argmax(y, dim=1)

    # Find indices for each class
    class_node_indices_0 = torch.where(class_labels == 0)[0]
    class_node_indices_1 = torch.where(class_labels == 1)[0]
    class_node_indices_2 = torch.where(class_labels == 2)[0]

    # Calculate average attention weights for each class
    attention_weights_class_0 = attention_weights[class_node_indices_0][:, head_idx].mean()
    attention_weights_class_1 = attention_weights[class_node_indices_1][:, head_idx].mean()
    attention_weights_class_2 = attention_weights[class_node_indices_2][:, head_idx].mean()

    
    avg_attention_weights = [
        attention_weights_class_0,
        attention_weights_class_1,
        attention_weights_class_2
    ]
    classes = [f'unknown ( %{attention_weights_class_0 : .2f})', f'allosteric( %{attention_weights_class_1 : .2f})', f'orthosteric (%{attention_weights_class_2 : .2f})']
    
    plt.figure(figsize=(10, 6))
    plt.bar(classes, avg_attention_weights, color=['orange', 'green', 'red'])
    plt.title(f'Average Attention Weights by Class (Layer {layer_idx}, Head {head_idx})')

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
    feature_names = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
                     "UNK", "phi", "psi", "omega", "tau", "theta", "atom_sasa", "alpha-helix", "beta-sheet", "coil", "x", "y", "z"]
    feature_importance = compute_node_feature_importance(model, data, num_samples=25, perturbation='permute')
    edge_features_importance = compute_edge_feature_importance(model, data, num_samples=25, perturbation='permute')
    edge_features_names = ["distance", "cosine_angle", "sequence"]
    plot_node_feature_importance(feature_importance, feature_names)
    plot_edge_feature_importance(edge_features_importance, edge_features_names)
    compute_saliency_maps(model, data, feature_names)
    average_attention_weights(model, data, layer_idx=0, head_idx=0)