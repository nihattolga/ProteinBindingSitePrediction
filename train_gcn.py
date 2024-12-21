import os
import time
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from models.models import GCN

# Constants
NODE_FEATURES_PATH = "biolip_node_features.npy"
NODE_LABELS_PATH = "biolip_node_labels.npy"
EDGE_INDEX_PATH = "biolip_edge_index.npy"
EDGE_FEATURES_PATH = "biolip_edge_features.npy"
BATCH_SIZE = 64
EPOCHS = 300
MODEL_PATH = "model_gcn.pth"
LEARNING_RATE = 0.01
EARLY_STOPPING_PATIENCE = 10  

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
writer = SummaryWriter()


def load_data():
    """Loads and preprocesses the data."""
    node_features = np.array(np.load(NODE_FEATURES_PATH, allow_pickle=True))
    node_labels = np.array(np.load(NODE_LABELS_PATH, allow_pickle=True))
    edge_index = np.array(np.load(EDGE_INDEX_PATH, allow_pickle=True))
    edge_features = np.array(np.load(EDGE_FEATURES_PATH, allow_pickle=True))

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
        if len(edge_index[i]) != 0 and edge_features[i].shape[0] != 0:
            nf = torch.tensor(node_features[i], dtype=torch.float)
            nl = torch.tensor(node_labels[i], dtype=torch.float)
            ei = torch.tensor(edge_index[i], dtype=torch.long).T
            ef = torch.tensor(edge_features[i], dtype=torch.float)
            pre_transform = T.Compose([T.GCNNorm(), T.ToSparseTensor()])
            d = Data(x=nf, edge_index=ei, edge_attr=ef, y=nl)
            d = pre_transform(d)
            datalist.append(d)
    
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
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    return train_loader, val_loader, test_loader


def initialize_model():
    """Initializes the model, loss function, and optimizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN(hidden_channels=256, num_layers=5, alpha=0.5, theta=1.0,
            shared_weights=False, dropout=0.2).to(device)
    class_weights = torch.tensor(1/cw, dtype=torch.float).to(device)
    pos_weight = torch.tensor(
        [
            (sum(cw) - cw[0]) / cw[0],
            (sum(cw) - cw[1]) / cw[1],
            (sum(cw) - cw[2]) / cw[2],
        ],
        dtype=torch.float,
    ).to(device)
    loss_op = torch.nn.BCEWithLogitsLoss(
        weight=class_weights, pos_weight=pos_weight
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return model, loss_op, optimizer, device


def train(model, loss_op, optimizer, device):
    """Trains the model for one epoch."""
    model.train()

    total_loss = total_examples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.adj_t)
        loss = loss_op(output, data.y)
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), MODEL_PATH)
    return total_loss / total_examples


@torch.no_grad()
def test(loader, model, device):
    """Evaluates the model on the given data loader."""
    model.eval()

    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        out = model(data.x.to(device), data.adj_t.to(device))
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    f1 = f1_score(y, pred, average="micro") if pred.sum() >= 0 else 0
    precision = (
        precision_score(y, pred, average="micro") if pred.sum() >= 0 else 0
    )
    recall = recall_score(y, pred, average="micro") if pred.sum() >= 0 else 0
    auc = (
        roc_auc_score(y, pred, average="micro", multi_class="ovr")
        if pred.sum() >= 0
        else 0
    )
    task_loss_1 = []
    task_loss_2 = []
    for node in range(len(pred)):
        if y[node][1]==1:
            if pred[node][1]==1:
                task_loss_1.append(0)
            else:
                task_loss_1.append(1)
        if y[node][2]==1:
            if pred[node][2]==1:
                task_loss_2.append(0)
            else:
                task_loss_2.append(1)

    task_loss_1 = np.array(task_loss_1).mean()
    task_loss_2 = np.array(task_loss_2).mean()
    task_loss = (task_loss_1 + task_loss_2)/2

    return (
        f1,
        precision,
        recall,
        auc,
        task_loss_1,
        task_loss_2,
        task_loss,
    )


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
        start = time.time()
        loss = train(model, loss_op, optimizer, device)
        (
            val_f1,
            val_precision,
            val_recall,
            val_auc,
            val_task_loss_1,
            val_task_loss_2,
            val_task_loss,
        ) = test(val_loader, model, device)
        (
            test_f1,
            test_precision,
            test_recall,
            test_auc,
            test_task_loss_1,
            test_task_loss_2,
            test_task_loss,
        ) = test(test_loader, model, device)
        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, / "
            f"Val: F1={val_f1:.4f}, Precision={val_precision:.4f}, Recall={val_recall:.4f}, AUC={val_auc:.4f}, Task Loss={val_task_loss:.4f}, Task Loss 1={val_task_loss_1:.4f}, Task Loss 2={val_task_loss_2:.4f}  / "
            f"Test: F1={test_f1:.4f}, Precision={test_precision:.4f}, Recall={test_recall:.4f}, AUC={test_auc:.4f}, Task Loss={test_task_loss:.4f}, Task Loss 1={test_task_loss_1:.4f}, Task Loss 2={test_task_loss_2:.4f}"
        )
        times.append(time.time() - start)

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)
        writer.add_scalar("Precision/val", val_precision, epoch)
        writer.add_scalar("Recall/val", val_recall, epoch)
        writer.add_scalar("AUC/val", val_auc, epoch)
        writer.add_scalar("Task Loss/val", val_task_loss, epoch)
        writer.add_scalar("Task Loss 1/val", val_task_loss_1, epoch)
        writer.add_scalar("Task Loss 2/val", val_task_loss_2, epoch)
        writer.add_scalar("F1/test", test_f1, epoch)
        writer.add_scalar("Precision/test", test_precision, epoch)
        writer.add_scalar("Recall/test", test_recall, epoch)
        writer.add_scalar("AUC/test", test_auc, epoch)
        writer.add_scalar("Task Loss/test", test_task_loss, epoch)
        writer.add_scalar("Task Loss 1/test", test_task_loss_1, epoch)
        writer.add_scalar("Task Loss 2/test", test_task_loss_2, epoch)

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
