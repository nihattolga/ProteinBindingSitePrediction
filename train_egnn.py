import os
import time
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss
)

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from models.models import EGNN

# Constants
NODE_FEATURES_PATH = "biolip_node_features.npy"
NODE_LABELS_PATH = "biolip_node_labels.npy"
EDGE_INDEX_PATH = "biolip_edge_index.npy"
EDGE_FEATURES_PATH = "biolip_edge_features.npy"
PDB_ID_PATH = "biolip_pdb_id.npy"
CHAIN_ID_PATH = "biolip_chain_id.npy"
LIG_COORDS_PATH = "biolip_lig_coords.npy"
LIG_NAMES_PATH = "biolip_lig_names.npy"
MODEL_PATH = "model_egnn_biolip.pth"

BATCH_SIZE = 64
EPOCHS = 300
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
writer = SummaryWriter()


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

    # Calculate class weights
    w = [np.sum(graph_labels, axis=0) for graph_labels in node_labels]
    weights = sum(w)
    cw = [
        sum(weights) / weights[i] if weights[i] != 0 else 0
        for i in range(len(weights))
    ]
    cw = max(cw) / cw

    # Create datalist
    datalist = []
    for i in range(len(node_features)):
        if len(edge_index[i]) != 0 and lig_coords[i].shape[0] != 0 and edge_features[i].shape[0] != 0:
            nf = torch.tensor(node_features[i][:,:-3], dtype=torch.float)
            nl = torch.tensor(node_labels[i], dtype=torch.float)
            ei = torch.tensor(edge_index[i], dtype=torch.long).T
            ef = torch.tensor(edge_features[i], dtype=torch.float)
            pdb_id = pdb_ids[i]
            chain_id = chain_ids[i]
            coords = torch.tensor(node_features[i][:,-3:], dtype=torch.float)
            l_coords = torch.tensor(lig_coords[i], dtype=torch.float)
            lig_name = lig_names[i]
            coords_label = closest_lig_coords(nl, l_coords, coords)
            datalist.append(Data(x=nf, edge_index=ei, edge_attr=ef, y=nl, coords=coords, coords_label=coords_label, pdb_id=pdb_id, chain_id=chain_id, lig_coords=l_coords, lig_name=lig_name))
    indices_with_empty_lig_coords = []

    for i, data in enumerate(datalist):
        if data.lig_coords.size(0) == 0:
            indices_with_empty_lig_coords.append(i)

    filtered_datalist = [data for i, data in enumerate(datalist) if i not in indices_with_empty_lig_coords]

    return filtered_datalist, cw

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

def split_data(datalist):
    """Splits the data into train, validation, and test sets."""
    data_size = len(datalist)
    train_size = int(0.8 * data_size)
    val_size = int(0.1 * data_size)
    test_size = data_size - train_size - val_size

    train_dataset = datalist[:train_size]
    val_dataset = datalist[train_size:train_size + val_size]
    test_dataset = datalist[train_size + val_size:]

    print(f"Total data size: {data_size}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset):
    """Creates data loaders for train, validation, and test sets."""
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    return train_loader, val_loader, test_loader


def initialize_model():
    """Initializes the model, loss function, and optimizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EGNN(in_node_nf=30, hidden_nf=256, out_node_nf=3, in_edge_nf=3, device=device, attention=True).to(device)
    class_weights = torch.tensor(cw, dtype=torch.float).to(device)
    pos_weight = torch.tensor(
        [
            (sum(cw) - cw[0]) / cw[0],
            (sum(cw) - cw[1]) / cw[1],
            (sum(cw) - cw[2]) / cw[2],
        ],
        dtype=torch.float,
    ).to(device)
    loss_op = torch.nn.CrossEntropyLoss(
        ignore_index=0
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return model, loss_op, optimizer, device


def train(model, loss_op, optimizer, device):
    """Trains the model for one epoch."""
    model.train()

    total_loss = total_samples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        h, x = model(h=data.x, x=data.coords,edges=data.edge_index, edge_attr=data.edge_attr)
        
        mask = torch.any(data.coords_label != 0, dim=1)

        coord_loss = torch.nn.MSELoss(reduction="sum")(x[mask], data.coords_label[mask])
        labels = torch.argmax(data.y, dim=1) 
        binding_loss = loss_op(h, labels)
        loss = binding_loss + coord_loss
        total_loss += loss.item() * data.num_nodes 
        total_samples += data.num_nodes * data.coords.shape[0] * data.coords.shape[1]
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), MODEL_PATH)
    return total_loss / total_samples


@torch.no_grad()
def test(model, loader, device):
    """Evaluates the model, calculating metrics for known sites only."""
    model.eval()
    all_labels, all_probs = [], []  
    all_coords_labels, all_coords_preds = [], []

    for data in loader:
        data = data.to(device)
        logits, lig_coords_pred = model(h=data.x, x=data.coords,edges=data.edge_index, edge_attr=data.edge_attr)
        probs = F.softmax(logits, dim=1)
        labels = data.y 

        all_coords_labels.append(data.coords_label.cpu())
        all_coords_preds.append(lig_coords_pred.cpu())
        
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    if all_labels:
        all_coords_labels = torch.cat(all_coords_labels)
        all_coords_preds = torch.cat(all_coords_preds)

        coords_mask = torch.any(all_coords_labels != 0, dim=1)
        coord_loss = F.mse_loss(all_coords_preds[coords_mask], all_coords_labels[coords_mask], reduction = "sum") / coords_mask.sum()

        y_true = torch.cat(all_labels).numpy()
        y_probs = torch.cat(all_probs).numpy() 

         # Get integer labels for non-probability metrics
        y_true_int = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_probs, axis=1)

        # Mask out non-binding sites
        mask = y_true_int != 0
        y_true_masked = y_true_int[mask]
        y_pred_masked = y_pred[mask]
        y_probs_masked = y_probs[mask]  
        metrics = {
            "f1": f1_score(y_true_masked, y_pred_masked, average="macro"),
            "precision": precision_score(y_true_masked, y_pred_masked, average="macro"),
            "recall": recall_score(y_true_masked, y_pred_masked, average="macro"),
            "auc": roc_auc_score(y_true[mask][:, 1:], y_probs_masked[:, 1:], average="macro", multi_class="ovr"),
            "brier": np.mean(np.sum((y_true[mask][:, 1:] - y_probs_masked[:, 1:])**2, axis=1)), 
            "log_loss": log_loss(y_true[mask], y_probs_masked)  
        }

        y_pred_masked = np.argmax(y_probs_masked[:, 1:], axis=1) + 1 
        y_true_masked_onehot = y_true[mask][:,1:] 
        y_true_masked = np.argmax(y_true_masked_onehot, axis=1) +1 
        task_loss_1 = np.mean((y_true_masked == 1) != (y_pred_masked == 1)) 
        task_loss_2 = np.mean((y_true_masked == 2) != (y_pred_masked == 2)) 
        task_loss = (task_loss_1 + task_loss_2)/2
        metrics["task_loss_1"] = task_loss_1
        metrics["task_loss_2"] = task_loss_2
        metrics["task_loss"] = task_loss
        metrics["coord_loss"] = coord_loss.item()
    else:
        metrics = {'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0, 'brier': 0, 'log_loss': 0, 'task_loss_1': 0, 'task_loss_2': 0, 'task_loss': 0, 'coord_loss': 0}

    return metrics


if __name__ == "__main__":
    datalist, cw = load_data()
    train_dataset, val_dataset, test_dataset = split_data(datalist)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset
    )
    model, loss_op, optimizer, device = initialize_model()

    scheduler = MultiStepLR(optimizer, milestones=[75,150,225], gamma=0.1)

    times = []
    best_val_f1 = 0
    epochs_without_improvement = 0
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, loss_op, optimizer, device)
       
        val_metrics = test(model, val_loader, device)
        test_metrics = test(model, test_loader, device)

        print(f"Epoch: {epoch:03d}, Train Loss: {loss:.4f}")
        print(
            f"Val F1: {val_metrics['f1']:.4f}, Val Precision: {val_metrics['precision']:.4f}, "
            f"Val Recall: {val_metrics['recall']:.4f}, Val AUC: {val_metrics['auc']:.4f}, "
            f"Val Brier: {val_metrics['brier']:.4f}, Val Log Loss: {val_metrics['log_loss']:.4f}, "
            f"Val Task Loss: {val_metrics['task_loss']:.4f}, Val Task Loss 1: {val_metrics['task_loss_1']:.4f}, "
            f"Val Task Loss 2: {val_metrics['task_loss_2']:.4f}, Val Coord Loss: {val_metrics['coord_loss']:.4f}"
        )
        print(
              f"Test F1: {test_metrics['f1']:.4f}, Test Precision: {test_metrics['precision']:.4f}, "
              f"Test Recall: {test_metrics['recall']:.4f}, Test AUC: {test_metrics['auc']:.4f}, "
              f"Test Brier: {test_metrics['brier']:.4f}, Test Log Loss: {test_metrics['log_loss']:.4f}, "
              f"Test Task Loss: {test_metrics['task_loss']:.4f}, "
              f"Test Task Loss 1: {test_metrics['task_loss_1']:.4f}, Test Task Loss 2: {test_metrics['task_loss_2']:.4f}"
              f"Test Coord Loss: {test_metrics['coord_loss']:.4f}"
        )

        # TensorBoard logging
        writer.add_scalar("Loss/train", loss, epoch)

        for metric_name, metric_value in val_metrics.items():
            writer.add_scalar(f"{metric_name.title()}/Val", metric_value, epoch)

        for metric_name, metric_value in test_metrics.items():
            writer.add_scalar(f"{metric_name.title()}/Test", metric_value, epoch)

        """# Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break"""
        scheduler.step()
    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

    writer.close()
