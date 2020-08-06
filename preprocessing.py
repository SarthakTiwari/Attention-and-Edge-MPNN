import numpy as np
import torch
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from collections import defaultdict


def one_of_k_encoding(x, allowable_set):
  if x not in allowable_set:
    raise Exception("input {0} not in allowable set{1}:".format(
        x, allowable_set))
  return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
  """Maps inputs not in the allowable set to the last element."""
  if x not in allowable_set:
    x = allowable_set[-1]
  return list(map(lambda s: x == s, allowable_set))



def atom_features(atom,
                  bool_id_feat=False,
                  explicit_H=False):
  if bool_id_feat:
    return np.array([atom_to_id(atom)])
  else:
    results = one_of_k_encoding_unk(
      atom.GetSymbol(),
      [
        'C',
        'N',
        'O',
        'S',
        'F',
        'Si',
        'P',
        'Cl',
        'Br',
        'I',
        'H',  # H?
        'Unknown'
      ]) + one_of_k_encoding(atom.GetDegree(),
                             [0, 1, 2, 3, 4, 5, 6, 7]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
              ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
      results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                [0, 1, 2, 3, 4])

    return np.array(results)


BONDTYPE_TO_INT = defaultdict(
    lambda: 0,
    {
        BondType.SINGLE: 0,
        BondType.DOUBLE: 1,
        BondType.TRIPLE: 2,
        BondType.AROMATIC: 3
    }
)

def smile_to_graph(smile):
    molecule = Chem.MolFromSmiles(smile)
    n_atoms = molecule.GetNumAtoms()
    atoms = [molecule.GetAtomWithIdx(i) for i in range(n_atoms)]

    adjacency = Chem.rdmolops.GetAdjacencyMatrix(molecule)
    node_features = np.array([atom_features(atom) for atom in atoms])

    n_edge_features = 4
    edge_features = np.zeros([n_atoms, n_atoms, n_edge_features])
    for bond in molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = BONDTYPE_TO_INT[bond.GetBondType()]
        edge_features[i, j, bond_type] = 1
        edge_features[j, i, bond_type] = 1

    return adjacency, node_features, edge_features


def molgraph_collate_fn(data):

    n_samples = len(data)
    if len(data[0])==2:
  
      (adjacency_0, node_features_0, edge_features_0), targets_0 = data[0]
      n_nodes_largest_graph = max(map(lambda sample: sample[0][0].shape[0], data))
      n_node_features = node_features_0.shape[1]
      n_edge_features = edge_features_0.shape[2]
      n_targets = len(targets_0)

      adjacency_tensor = torch.zeros(n_samples, n_nodes_largest_graph, n_nodes_largest_graph)
      node_tensor = torch.zeros(n_samples, n_nodes_largest_graph, n_node_features)
      edge_tensor = torch.zeros(n_samples, n_nodes_largest_graph, n_nodes_largest_graph, n_edge_features)
      target_tensor = torch.zeros(n_samples,n_targets)

      for i in range(n_samples):
          (adjacency, node_features, edge_features), target = data[i]
          n_nodes = adjacency.shape[0]

          adjacency_tensor[i, :n_nodes, :n_nodes] = torch.Tensor(adjacency)
          node_tensor[i, :n_nodes, :] = torch.Tensor(node_features)
          edge_tensor[i, :n_nodes, :n_nodes, :] = torch.Tensor(edge_features)

          target_tensor[i] = torch.Tensor(target)

      return adjacency_tensor, node_tensor, edge_tensor, target_tensor

    else:
      (adjacency_0, node_features_0, edge_features_0) = data[0]
      n_nodes_largest_graph = max(map(lambda sample: sample[0][0].shape[0], data))
      n_node_features = node_features_0.shape[1]
      n_edge_features = edge_features_0.shape[2]


      adjacency_tensor = torch.zeros(n_samples, n_nodes_largest_graph, n_nodes_largest_graph)
      node_tensor = torch.zeros(n_samples, n_nodes_largest_graph, n_node_features)
      edge_tensor = torch.zeros(n_samples, n_nodes_largest_graph, n_nodes_largest_graph, n_edge_features)

      for i in range(n_samples):
          (adjacency, node_features, edge_features) = data[i]
          n_nodes = adjacency.shape[0]

          adjacency_tensor[i, :n_nodes, :n_nodes] = torch.Tensor(adjacency)
          node_tensor[i, :n_nodes, :] = torch.Tensor(node_features)
          edge_tensor[i, :n_nodes, :n_nodes, :] = torch.Tensor(edge_features)


      return adjacency_tensor, node_tensor, edge_tensor
