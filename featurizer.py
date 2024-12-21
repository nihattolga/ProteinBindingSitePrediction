import biotite.structure.io as strucio
import biotite.structure as struc
import biotite.database.rcsb as rcsb

from tempfile import gettempdir

import numpy as np
import math

class Featurizer:
    """
    A class to featurize protein structures for graph neural networks.
    """

    def __init__(self, pdb_id, chain_id, allo_res, ortho_res, lig_names):
        """
        Initializes the Featurizer class.

        Args:
            casbench_path (str): Path to the CASBench directory.
        """
        self.pdb_id = pdb_id
        self.chain_id = chain_id
        self.allo_res = allo_res
        self.ortho_res = ortho_res
        self.lig_names = lig_names


    def compute(self):
        """
        Computes the node features, node labels, edge index, and edge features.

        Returns:
            tuple: A tuple containing the node features, node labels, edge index,
                and edge features.
        """
        file_path = rcsb.fetch(self.pdb_id, "pdb", gettempdir())
        protein_struc = strucio.load_structure(file_path)
        filtered_struc = protein_struc[protein_struc.chain_id == self.chain_id]
        lig_struc = filtered_struc[np.isin(filtered_struc.res_name, self.lig_names)]
        lig_coords = lig_struc.coord
        lig_name = lig_struc.res_name
        ca_array = self.get_features(protein_struc)
        ca_array = ca_array[ca_array.chain_id == self.chain_id]
        node_features, node_labels = self.get_one_hot(ca_array)
        edge_index, edge_features = self.get_edges(ca_array) 

        return node_features, node_labels, edge_index, edge_features, self.pdb_id, self.chain_id, lig_coords, lig_name

    def get_edges(self, ca_array):
        """
        Computes the edge index and edge features.

        Args:
            ca_array (biotite.structure.AtomArray): Atom array containing
                the protein structure.

        Returns:
            tuple: A tuple containing the edge index and edge features.
        """
        edge_features, edge_index = [], []
        res_ids = ca_array.res_id
        for i in range(len(res_ids) - 1):
            # Check if both residue i and i+1 exist
            if (res_ids == res_ids[i]).any() and (
                res_ids == res_ids[i + 1]
            ).any():
                edge_1 = np.where(res_ids == res_ids[i])[0][0]
                edge_2 = np.where(res_ids == res_ids[i + 1])[0][0]
                d = math.dist(ca_array[edge_1].coord, ca_array[edge_2].coord)
                cos_angle = (
                    np.dot(ca_array[edge_1].coord, ca_array[edge_2].coord)
                    / np.linalg.norm(ca_array[edge_1].coord)
                    / np.linalg.norm(ca_array[edge_2].coord)
                )
                edge_index.append([edge_1, edge_2])
                edge_index.append([edge_2, edge_1])
                edge_features.append([d, cos_angle, 1])
                edge_features.append([d, cos_angle, -1])

        return edge_index, edge_features

    def get_one_hot(self, atom_array):
        """
        Computes the one-hot encoded node features and node labels.

        Args:
            atom_array (biotite.structure.AtomArray): Atom array containing
                the protein structure.

        Returns:
            tuple: A tuple containing the one-hot encoded node features and
                node labels.
        """
        node_features = {
            "ALA": 0,
            "ARG": 1,
            "ASN": 2,
            "ASP": 3,
            "CYS": 4,
            "GLU": 5,
            "GLN": 6,
            "GLY": 7,
            "HIS": 8,
            "ILE": 9,
            "LEU": 10,
            "LYS": 11,
            "MET": 12,
            "PHE": 13,
            "PRO": 14,
            "SER": 15,
            "THR": 16,
            "TRP": 17,
            "TYR": 18,
            "VAL": 19,
            "UNK": 20,
            "phi": 21,
            "psi": 22,
            "omega": 23,
            "tau": 24,
            "theta": 25,
            "atom_sasa": 26,
            "a": 27,
            "b": 28,
            "c": 29,
            "x": 30,
            "y": 31,
            "z": 32,
        }

        node_labels = {
            "not_site": 0,
            "allosteric": 1,
            "orthosteric": 2,
        }

        nf = []
        nl = []
        for atom in atom_array:
            one_hot_node = np.zeros(len(node_features), dtype=float)
            one_hot_labels = np.zeros(len(node_labels), dtype=float)
            one_hot_node[node_features[atom.res_name]] = 1.0
            one_hot_node[node_features["phi"]] = (
                atom.phi if atom.phi and ~np.isnan(atom.phi) else 0
            )
            one_hot_node[node_features["psi"]] = (
                atom.psi if atom.psi and ~np.isnan(atom.psi) else 0
            )
            one_hot_node[node_features["omega"]] = (
                atom.omega if atom.omega and ~np.isnan(atom.omega) else 0
            )
            one_hot_node[node_features["tau"]] = (
                atom.tau if atom.tau and ~np.isnan(atom.tau) else 0
            )
            one_hot_node[node_features["theta"]] = (
                atom.theta if atom.theta and ~np.isnan(atom.theta) else 0
            )
            one_hot_node[node_features["atom_sasa"]] = (
                atom.atom_sasa
                if atom.atom_sasa and ~np.isnan(atom.atom_sasa)
                else 0
            )
            one_hot_node[node_features[atom.sse]] = 1.0
            one_hot_node[node_features["x"]] = atom.coord[0]
            one_hot_node[node_features["y"]] = atom.coord[1]
            one_hot_node[node_features["z"]] = atom.coord[2]
            one_hot_labels[node_labels["not_site"]] = 1.0

            # Check for site labels
            for i in self.allo_res:
                if atom.res_id == i:
                    one_hot_labels[node_labels["allosteric"]] = 1.0
                    one_hot_labels[node_labels["not_site"]] = 0
            for i in self.ortho_res:
                if atom.res_id == i:
                    one_hot_labels[node_labels["orthosteric"]] = 1.0
                    one_hot_labels[node_labels["not_site"]] = 0

            nf.append(one_hot_node)
            nl.append(one_hot_labels)
        return nf, nl

    def get_features(self, atom_array):
        """
        Computes the features for each CA atom in the protein structure.

        Args:
            atom_array (biotite.structure.AtomArray): Atom array containing
                the protein structure.

        Returns:
            biotite.structure.AtomArray: Atom array containing only the CA
                atoms with computed features.
        """
        chains = np.unique(atom_array.chain_id)
        phi, psi, omega, atom_sasa, sse = self.get_features_vectorized(
            atom_array
        )

        atom_array.set_annotation("atom_sasa", atom_sasa)
        ca_array = atom_array[
            (atom_array.atom_name == "CA") & (atom_array.hetero == False)
        ]
        theta, tau = [], []
        for chain in chains:
            theta.extend(
                self.get_theta(ca_array[ca_array.chain_id == chain], i)
                for i in range(len(ca_array[ca_array.chain_id == chain]))
            )
            tau.extend(
                self.get_tau(ca_array[ca_array.chain_id == chain], i)
                for i in range(len(ca_array[ca_array.chain_id == chain]))
            )
        ca_array.set_annotation("phi", phi)
        ca_array.set_annotation("psi", psi)
        ca_array.set_annotation("omega", omega)
        ca_array.set_annotation("tau", tau)
        ca_array.set_annotation("theta", theta)
        ca_array.set_annotation("sse", sse)

        return ca_array

    def get_tau(self, atom_array, i):
        """
        Computes the tau angle for a given CA atom.

        Args:
            atom_array (biotite.structure.AtomArray): Atom array containing
                the protein structure.
            i (int): Index of the CA atom.

        Returns:
            float: The tau angle.
        """
        if i + 4 >= len(atom_array):
            return None
        return struc.dihedral(
            atom_array[i], atom_array[i + 1], atom_array[i + 2], atom_array[i + 4]
        )

    def get_theta(self, atom_array, i):
        """
        Computes the theta angle for a given CA atom.

        Args:
            atom_array (biotite.structure.AtomArray): Atom array containing
                the protein structure.
            i (int): Index of the CA atom.

        Returns:
            float: The theta angle.
        """
        if i + 3 >= len(atom_array):
            return None
        return struc.angle(atom_array[i], atom_array[i + 1], atom_array[i + 2])

    def get_dihedral_angles(self, atom_array):
        """
        Computes the phi, psi, and omega dihedral angles.

        Args:
            atom_array (biotite.structure.AtomArray): Atom array containing
                the protein structure.

        Returns:
            tuple: A tuple containing the phi, psi, and omega angles.
        """
        phi, psi, omega = struc.dihedral_backbone(atom_array)
        return phi, psi, omega

    def get_sasa(self, atom_array):
        """
        Computes the solvent accessible surface area (SASA) for each atom.

        Args:
            atom_array (biotite.structure.AtomArray): Atom array containing
                the protein structure.

        Returns:
            np.ndarray: Array containing the SASA for each atom.
        """
        atom_sasa = struc.sasa(atom_array, vdw_radii="Single")
        return atom_sasa

    def get_sse(self, atom_array):
        """
        Annotates the secondary structure elements (SSE) for each atom.

        Args:
            atom_array (biotite.structure.AtomArray): Atom array containing
                the protein structure.

        Returns:
            np.ndarray: Array containing the SSE annotation for each atom.
        """
        return struc.annotate_sse(atom_array)

    def get_features_vectorized(self, atom_array):
        """
        Computes the phi, psi, omega, SASA, and SSE features for all atoms.

        Args:
            atom_array (biotite.structure.AtomArray): Atom array containing
                the protein structure.

        Returns:
            tuple: A tuple containing the phi, psi, omega, SASA, and SSE
                features.
        """
        phi, psi, omega = struc.dihedral_backbone(atom_array)
        l = len(
            atom_array[(atom_array.atom_name == "CA") & (atom_array.hetero == False)]
        )
        if not len(phi) == l:
            phi, psi, omega = struc.dihedral_backbone(
                atom_array[(atom_array.hetero == False)]
            )
        atom_sasa = struc.sasa(atom_array, vdw_radii="Single")
        sse = struc.annotate_sse(atom_array)
        if not len(sse) == l:
            sse = struc.annotate_sse(atom_array[(atom_array.hetero == False)])
        return phi, psi, omega, atom_sasa, sse[sse != ""]