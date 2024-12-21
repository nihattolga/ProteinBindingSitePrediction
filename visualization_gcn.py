import numpy as np
from Bio.PDB import PDBParser, PDBIO, is_aa
from featurizer import Featurizer
import torch
from torch_geometric.data import Data, Batch
import torch_geometric.transforms as T
from models.models import GCN
from sklearn.cluster import DBSCAN


MODEL_PATH = "model_gcn_ss_biolip.pth" 

device = torch.device("cpu")

def initialize_model():
    """Initializes the GAT model."""
    model = GCN(hidden_channels=256, num_layers=2, alpha=0.5, theta=1.0,
            shared_weights=False, dropout=0.2).to(device)
    return model

def get_site_indices(model, data):
    """
    Gets the indices of the predicted allosteric and catalytic sites.

    Args:
        model (torch.nn.Module): The trained model.
        data (torch_geometric.data.Data): The input data.

    Returns:
        tuple: A tuple containing the indices of the allosteric and catalytic sites.
    """
    x, adj_t = data.x.clone(), data.adj_t.clone()
    with torch.no_grad():
        out = model(x, adj_t).detach()
        probs = torch.sigmoid(out).cpu().numpy()  
        probs = probs*500
        # Get indices of nodes where node[1] == 1 (allo_sites)
        allo_sites = (out[:, 1] > 0).nonzero().squeeze().cpu().numpy()

        # Get indices of nodes where node[2] == 1 (ortho_sites)
        ortho_sites = (out[:, 2] > 0).nonzero().squeeze().cpu().numpy()

        allo_probs = probs[:, 1]
        ortho_probs = probs[:, 2]

    return allo_sites, ortho_sites, allo_probs, ortho_probs

def save_binding_sites_to_pdb(structure, site_indices, filename):
    """
    Saves the binding site atoms to a PDB file.

    Args:
        structure (Bio.PDB.Structure): The original protein structure.
        site_indices: The indices of the atoms in the binding site.
        filename (str): The output PDB file name.
    """
    class BindingSiteSelect:
        def __init__(self, site_indices):
            self.site_indices = set(site_indices)

        def accept_atom(self, atom):
            return True

        def accept_model(self, model):
            return True

        def accept_chain(self, chain):
            return True

        def accept_residue(self, residue):
            return residue.get_id()[1] in self.site_indices

        def accept_atom(self, atom):
            return True

    io = PDBIO()
    io.set_structure(structure)
    io.save(filename, select=BindingSiteSelect(site_indices))

def assign_b_factors(structure, output_pdb, b_factors):
    
    protein_residues = []
    for residue in structure.get_residues():
        # Exclude heteroatoms and water (e.g., HOH)
        if residue.id[0] == " " and is_aa(residue, standard=True):
            protein_residues.append(residue)

    # Ensure number of residues matches probabilities
    if len(protein_residues) != len(b_factors):
        raise ValueError("Number of residues and B-factor values must match!")

    # Assign B-factors
    for residue, b_factor in zip(protein_residues, b_factors):
        for atom in residue:
            atom.bfactor = b_factor

    # Save modified PDB
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)

def cluster_predictions(probs, coords, eps=5, min_samples=5):  
    """Clusters predicted probabilities based on 3D coordinates.

    Args:
        probs: 1D array/tensor of predicted probabilities.
        coords: 2D array/tensor of 3D coordinates.
        eps: The maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples: The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        A tuple containing cluster labels and high probability indices for each cluster.
    """

    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()
    
    high_prob_mask = probs >= np.mean(probs)
    
    filtered_coords = coords[high_prob_mask]
    filtered_probs = probs[high_prob_mask]

    if len(filtered_coords) < min_samples:  
        return None, None 
    
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean') 
    clusters = db.fit_predict(filtered_coords)
    print(clusters)

    high_prob_indices_per_cluster = []

    for cluster_id in np.unique(clusters):
        if cluster_id != -1: 
            cluster_mask = clusters == cluster_id
            # Get indices of the high probability sites within the current cluster
            high_prob_indices = np.where(high_prob_mask)[0][np.where(cluster_mask)[0]]
            high_prob_indices_per_cluster.append((cluster_id, high_prob_indices)) 


    return clusters, high_prob_indices_per_cluster

def cluster_predictions_ortho(probs, coords, eps=5, min_samples=5):

    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()
    high_prob_mask = probs >= np.mean(probs)
    
    filtered_coords = coords[high_prob_mask]
    filtered_probs = probs[high_prob_mask]

    if len(filtered_coords) < min_samples: 
        return None, None 

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean') 
    clusters = db.fit_predict(filtered_coords)
    print(clusters)

    max_prob_cluster = None
    max_avg_prob = -1
    high_prob_indices_per_cluster = []
    for cluster_id in np.unique(clusters):
        if cluster_id != -1:
            cluster_mask = clusters == cluster_id
            cluster_probs_in_cluster = filtered_probs[cluster_mask] #Probabilities of the sites in current cluster

            avg_prob_in_cluster = np.mean(cluster_probs_in_cluster)  # Average probability in the current cluster
            if avg_prob_in_cluster > max_avg_prob:
                max_avg_prob = avg_prob_in_cluster
                # Get indices of the high probability sites within the current cluster
                high_prob_indices = np.where(high_prob_mask)[0][np.where(cluster_mask)[0]]
                max_prob_cluster = (cluster_id, high_prob_indices)

    if max_prob_cluster is not None:
        return max_prob_cluster # return cluster id and indices
    else: return None

def assign_cluster_b_factors(structure, allo_clusters, allo_high_prob_indices, output_pdb):
    """Assigns B-factors based on allosteric cluster assignments.

    Args:
        structure: The Bio.PDB.Structure object.
        allo_clusters: Cluster labels from DBSCAN.
        allo_high_prob_indices: List of tuples, where each tuple contains (cluster_id, indices).
        output_pdb (str): Path to save the output PDB file.
    """
    protein_residues = []
    for residue in structure.get_residues():
        if residue.id[0] == " " and is_aa(residue, standard=True):
            protein_residues.append(residue)

    # Initialize B-factors to 0
    b_factors = np.zeros(len(protein_residues))

    if allo_high_prob_indices is not None:  

        for cluster_id, indices in allo_high_prob_indices:
            cluster_b_factor = (cluster_id + 1) * 10 
            b_factors[indices] = cluster_b_factor


    # Assign B-factors to the structure
    for residue, b_factor in zip(protein_residues, b_factors):
        for atom in residue:
            atom.bfactor = b_factor

    # Save the modified PDB structure
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)


if __name__ == "__main__":
    # Load the trained model
    model = initialize_model()
    model.load_state_dict(torch.load(MODEL_PATH))

    # Load the data
    protein_path = "2be9.pdb"
    f = Featurizer(protein_path)
    nf, ei, ef, r = f.compute()
    node_features = []
    edge_index = []
    edge_features = []
    res_id = []
    for i in range(len(nf)):
        node_features.append(nf[i])
        edge_index.append(ei[i])
        edge_features.append(ef[i])
        res_id.append(r[i])

    # Create datalist
    datalist = []
    for i in range(len(node_features)):
        nf = torch.tensor(node_features[i], dtype=torch.float)
        ei = torch.tensor(edge_index[i], dtype=torch.long).T
        ef = torch.tensor(edge_features[i], dtype=torch.float)
        r = torch.tensor(res_id[i], dtype=torch.int)
        pre_transform = T.Compose([T.GCNNorm(), T.ToSparseTensor()])
        d = Data(x=nf, edge_index=ei, edge_attr=ef, res_id=r)
        d = pre_transform(d)
        datalist.append(d)

    data = Batch.from_data_list(datalist)
    
    # Get the predicted site indices
    allo_sites, ortho_sites, allo_probs, ortho_probs = get_site_indices(model, data)
    ortho_probs = np.power(ortho_probs/500, 3) 
    ortho_probs = (ortho_probs - ortho_probs.min()) / (ortho_probs.max() - ortho_probs.min()) * 100
    # Load the PDB structure
    parser = PDBParser()
    structure = parser.get_structure("protein", protein_path)

    # Get the unique residue IDs for the predicted sites
    allo_indices = np.unique(data.res_id[allo_sites])
    ortho_indices = np.unique(data.res_id[ortho_sites])

    # Save the binding sites to PDB files
    save_binding_sites_to_pdb(structure, ortho_sites, f"{protein_path[:4]}_ortho_sites.pdb")
    save_binding_sites_to_pdb(structure, allo_sites, f"{protein_path[:4]}_allo_sites.pdb")

    assign_b_factors(structure, f"{protein_path[:4]}_ortho_probs.pdb", ortho_probs)
    assign_b_factors(structure, f"{protein_path[:4]}_allo_probs.pdb", allo_probs)

    allo_coords = data.x[:, -3:].cpu().numpy()  
    ortho_coords = data.x[:, -3:].cpu().numpy()

    allo_clusters, allo_high_prob_indices = cluster_predictions(allo_probs, allo_coords, eps=5, min_samples=2) 
    assign_cluster_b_factors(structure, allo_clusters, allo_high_prob_indices, f"{protein_path[:4]}_allo_cluster_bfactors.pdb")
    ortho_clusters, ortho_high_prob_indices = cluster_predictions(ortho_probs, ortho_coords, eps=5, min_samples=2)
    assign_cluster_b_factors(structure, ortho_clusters, ortho_high_prob_indices, f"{protein_path[:4]}_ortho_cluster_bfactors.pdb")

    if allo_high_prob_indices:
        allo_all_indices = np.concatenate([indices for _, indices in allo_high_prob_indices])
        save_binding_sites_to_pdb(structure, allo_all_indices, f"{protein_path[:4]}_allo_all_clusters.pdb") 
        print(f"Allosteric cluster indices: {allo_all_indices}")
    else:
        print("No allosteric clusters found.")


    ortho_cluster, ortho_highest_prob_indices = cluster_predictions_ortho(ortho_probs, ortho_coords, eps=5, min_samples=2)

    if ortho_highest_prob_indices is not None:
        save_binding_sites_to_pdb(structure, ortho_highest_prob_indices, f"{protein_path[:4]}_ortho_highest_prob_cluster.pdb")
        print(f"Ortho highest probability cluster): {ortho_highest_prob_indices}")
    else:
        print("No orthosteric clusters found.")

    

