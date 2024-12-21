import pandas as pd
import numpy as np
from featurizer import Featurizer

df = pd.read_csv("biolip_nr_egnn.csv")
node_features = []
node_labels = []
edge_index = []
edge_features = []
pdb_ids = []
chain_ids = []
lig_coords = []
lig_names = []

for i in range(len(df)):
    pdb_id = df.iloc[i][0]
    chain_id = df.iloc[i][1]
    l_names = df.iloc[i][2].split()
    ortho_res = np.array(
        [int(x) for x in df.iloc[i][3].split(',')]
    )
    allo_res = np.array(
        [int(x) for x in df.iloc[i][4].split(',')]
    )

    try:
        nf, nl, ei, ef, pdb_id, chain_id, lig_coord, lig_name = Featurizer(pdb_id, chain_id, allo_res, ortho_res, l_names).compute()
        node_features.append(np.array(nf))
        node_labels.append(np.array(nl))
        edge_index.append(np.array(ei))
        edge_features.append(np.array(ef))
        pdb_ids.append(pdb_id)
        chain_ids.append(chain_id)
        lig_coords.append(lig_coord)
        lig_names.append(lig_name)
        print(f"Processed {pdb_id}")
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        continue

# Save features to NumPy arrays
np.save("biolip_node_features.npy", np.array(node_features, dtype=object))
np.save("biolip_node_labels.npy", np.array(node_labels, dtype=object))
np.save("biolip_edge_index.npy", np.array(edge_index, dtype=object))
np.save("biolip_edge_features.npy", np.array(edge_features, dtype=object))
np.save("biolip_pdb_id.npy", np.array(pdb_ids, dtype=object))
np.save("biolip_chain_id.npy", np.array(chain_ids, dtype=object))
np.save("biolip_lig_coords.npy", np.array(lig_coords, dtype=object))
np.save("biolip_lig_names.npy", np.array(lig_names, dtype=object))

print("BioLip features saved to NumPy arrays.")