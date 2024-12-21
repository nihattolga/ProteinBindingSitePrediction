import os
import time
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, log_loss
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from models.models import GAT

# Constants
NODE_FEATURES_PATH = "biolip_node_features.npy"
NODE_LABELS_PATH = "biolip_node_labels.npy"
EDGE_INDEX_PATH = "biolip_edge_index.npy"
EDGE_FEATURES_PATH = "biolip_edge_features.npy"
MODEL_PATH = "model_gat_ss_biolip.pth"
BATCH_SIZE = 64
EPOCHS = 300
LEARNING_RATE = 0.01
EARLY_STOPPING_PATIENCE = 10

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
writer = SummaryWriter()

def load_data():
    """Loads and preprocesses the data."""
    node_features = np.load(NODE_FEATURES_PATH, allow_pickle=True)
    node_labels = np.load(NODE_LABELS_PATH, allow_pickle=True)
    edge_index = np.load(EDGE_INDEX_PATH, allow_pickle=True)
    edge_features = np.load(EDGE_FEATURES_PATH, allow_pickle=True)

    w = [np.sum(graph_labels, axis=0) for graph_labels in node_labels]
    weights = sum(w)
    cw = [
        sum(weights) / weights[i] if weights[i] != 0 else 0
        for i in range(len(weights))
    ]
    cw = max(cw) / cw

    datalist = []
    for i in range(len(node_features)):
        if len(edge_index[i]) != 0 and edge_features[i].shape[0] != 0:
            nf = torch.tensor(node_features[i], dtype=torch.float)
            nl = torch.tensor(node_labels[i], dtype=torch.float)
            ei = torch.tensor(edge_index[i], dtype=torch.long).T
            ef = torch.tensor(edge_features[i], dtype=torch.float)
            datalist.append(Data(x=nf, edge_index=ei, edge_attr=ef, y=nl))
    
    return datalist, cw

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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    return train_loader, val_loader, test_loader

def initialize_model(cw):
    """Initializes the model, loss function, and optimizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAT(num_layers=2, in_channels=33, hidden_channels=256, out_channels=3, dropout=0.2).to(device)
    class_weights = torch.tensor(1/cw, dtype=torch.float).to(device)
    pos_weight = torch.tensor(
        [
            (sum(cw) - cw[0]) / cw[0],
            (sum(cw) - cw[1]) / cw[1],
            (sum(cw) - cw[2]) / cw[2],
        ],
        dtype=torch.float,
    ).to(device)
    loss_op = torch.nn.CrossEntropyLoss(weight=pos_weight, ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return model, loss_op, optimizer, device

def train(model, loss_op, optimizer, train_loader, device):
    """Trains the model for one epoch."""
    model.train()
    total_loss = total_samples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.edge_attr)
        labels = torch.argmax(data.y, dim=1)
        loss = loss_op(logits, labels)
        mask = (labels == 1) | (labels == 2)
        total_loss += loss.item() * mask.sum().item() 
        total_samples += mask.sum().item() 
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), MODEL_PATH)
    return total_loss / total_samples

@torch.no_grad()
def test(model, loader, device):
    """Evaluates the model, calculating metrics for known sites only."""
    model.eval()
    all_labels, all_probs = [], []  
    for data in loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.edge_attr)
        probs = F.softmax(logits, dim=1)
        labels = data.y  
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    if all_labels: 
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
        y_true_masked_onehot = y_true[mask][:,1:] # Exclude non-binding sites from ground truth
        y_true_masked = np.argmax(y_true_masked_onehot, axis=1) +1 

        task_loss_1 = np.mean((y_true_masked == 1) != (y_pred_masked == 1)) 
        task_loss_2 = np.mean((y_true_masked == 2) != (y_pred_masked == 2)) 
        task_loss = (task_loss_1 + task_loss_2)/2
        metrics["task_loss_1"] = task_loss_1
        metrics["task_loss_2"] = task_loss_2
        metrics["task_loss"] = task_loss
    else:
        metrics = {'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0, 'brier': 0, 'log_loss': 0, 'task_loss_1': 0, 'task_loss_2': 0, 'task_loss': 0}

    return metrics


if __name__ == "__main__":
    datalist, class_weights = load_data()
    train_dataset, val_dataset, test_dataset = split_data(datalist)
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset)
    model, loss_op, optimizer, device = initialize_model(class_weights)

    best_val_f1 = 0
    patience_counter = 0

    scheduler = MultiStepLR(optimizer, milestones=[100,200,300,400], gamma=0.1)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, loss_op, optimizer, train_loader, device)
        val_metrics = test(model, val_loader, device)
        test_metrics = test(model, test_loader, device)

        print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}")
        print(
            f"Val F1: {val_metrics['f1']:.4f}, Val Precision: {val_metrics['precision']:.4f}, "
            f"Val Recall: {val_metrics['recall']:.4f}, Val AUC: {val_metrics['auc']:.4f}, "
            f"Val Brier: {val_metrics['brier']:.4f}, Val Log Loss: {val_metrics['log_loss']:.4f}, "
            f"Val Task Loss: {val_metrics['task_loss']:.4f}, Val Task Loss 1: {val_metrics['task_loss_1']:.4f}, "
            f"Val Task Loss 2: {val_metrics['task_loss_2']:.4f}"
        )
        print(
              f"Test F1: {test_metrics['f1']:.4f}, Test Precision: {test_metrics['precision']:.4f}, "
              f"Test Recall: {test_metrics['recall']:.4f}, Test AUC: {test_metrics['auc']:.4f}, "
              f"Test Brier: {test_metrics['brier']:.4f}, Test Log Loss: {test_metrics['log_loss']:.4f}, "
              f"Test Task Loss: {test_metrics['task_loss']:.4f}, "
              f"Test Task Loss 1: {test_metrics['task_loss_1']:.4f}, Test Task Loss 2: {test_metrics['task_loss_2']:.4f}"
        )

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("F1/val", val_metrics["f1"], epoch)
        writer.add_scalar("Precision/val", val_metrics["precision"], epoch)
        writer.add_scalar("Recall/val", val_metrics["recall"], epoch)
        writer.add_scalar("AUC/val", val_metrics["auc"], epoch)
        writer.add_scalar("Brier/val", val_metrics["brier"], epoch)
        writer.add_scalar("Log Loss/val", val_metrics["log_loss"], epoch)
        writer.add_scalar("Task Loss/val", val_metrics["task_loss"], epoch)
        writer.add_scalar("Task Loss 1/val", val_metrics["task_loss_1"], epoch)
        writer.add_scalar("Task Loss 2/val", val_metrics["task_loss_2"], epoch)
        writer.add_scalar("F1/test", test_metrics["f1"], epoch)
        writer.add_scalar("Precision/test", test_metrics["precision"], epoch)
        writer.add_scalar("Recall/test", test_metrics["recall"], epoch)
        writer.add_scalar("AUC/test", test_metrics["auc"], epoch)
        writer.add_scalar("Brier/test", test_metrics["brier"], epoch)
        writer.add_scalar("Log Loss/test", test_metrics["log_loss"], epoch)
        writer.add_scalar("Task Loss/test", test_metrics["task_loss"], epoch)
        writer.add_scalar("Task Loss 1/test", test_metrics["task_loss_1"], epoch)
        writer.add_scalar("Task Loss 2/test", test_metrics["task_loss_2"], epoch)
        

        scheduler.step()

        # Early stopping based on validation F1
        """if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break"""
