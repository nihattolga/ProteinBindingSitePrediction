import requests

def get_macromolecule_from_pdb(pdb_id):
  """Fetches the macromolecule name from the PDB API for a given PDB ID.

  Args:
    pdb_id: The PDB ID of the structure.

  Returns:
    A string containing the macromolecule name, or None if not found.
  """

  url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
  response = requests.get(url)

  if response.status_code == 200:
    data = response.json()
    try:
        macromolecule = data['rcsb_polymer_entity']['pdbx_description']
        return macromolecule
    except (IndexError, KeyError):
        return None
  else:
    print(f"Error fetching data for PDB ID {pdb_id}: {response.status_code}")
    return None
  
