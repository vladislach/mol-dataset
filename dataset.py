from rdkit import Chem
import pandas as pd
import torch
import os

from torch_geometric.utils import one_hot
from torch_geometric.data import (
    InMemoryDataset,
    Data,
    download_url,
    extract_zip,
)


x_map = {
    'atomic_num': [6, 7, 8, 9, 16, 17],
    'degree': [1, 2, 3, 4],
    'num_hs': [0, 1, 2, 3],
    'hybridization': ['SP', 'SP2', 'SP3']
}

e_map = {
    'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
}


def get_node_feats(rdmol: Chem.Mol) -> torch.Tensor:
    """Generates a feature tensor for each atom (node) in an RDKit molecule object (rdmol)."""
    feats = []

    feats.append(
        one_hot(
            torch.tensor([x_map['atomic_num'].index(atom.GetAtomicNum()) for atom in rdmol.GetAtoms()]),
            num_classes=len(x_map['atomic_num']), dtype=torch.float
        )
    )
    feats.append(
        one_hot(
            torch.tensor([x_map['degree'].index(atom.GetTotalDegree()) for atom in rdmol.GetAtoms()]),
            num_classes=len(x_map['degree']), dtype=torch.float
        )
    )
    feats.append(
        one_hot(
            torch.tensor([x_map['num_hs'].index(atom.GetTotalNumHs()) for atom in rdmol.GetAtoms()]),
            num_classes=len(x_map['num_hs']), dtype=torch.float
        )
    )
    feats.append(
        one_hot(
            torch.tensor([x_map['hybridization'].index(str(atom.GetHybridization())) for atom in rdmol.GetAtoms()]),
            num_classes=len(x_map['hybridization']), dtype=torch.float
        )
    )
    feats.append(
        torch.tensor([atom.GetIsAromatic() for atom in rdmol.GetAtoms()], dtype=torch.float).view(-1, 1)
    )
    for i in range(3, 7):
        feats.append(
            torch.tensor([atom.IsInRingSize(i) for atom in rdmol.GetAtoms()], dtype=torch.float).view(-1, 1)
        )
        
    return torch.cat(feats, dim=-1)

def get_edge_index(mol: Chem.Mol) -> torch.Tensor:
    """Generates an edge index tensor (edge connectivity) for an RDKit molecule object (mol)."""
    return torch.tensor(Chem.GetAdjacencyMatrix(mol), dtype=torch.long).to_sparse().indices()

def get_edge_attr(rdmol: Chem.Mol, edge_index: torch.Tensor) -> torch.Tensor:
    """Generates a feature tensor for each bond (edge) in an RDKit molecule object (rdmol)."""
    edge_attr = []

    for i, j in edge_index.T:
        bond = rdmol.GetBondBetweenAtoms(i.item(), j.item())
        edge_attr.append(e_map['bond_type'].index(str(bond.GetBondType())))
    
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)
    return one_hot(edge_attr, num_classes=len(e_map['bond_type']), dtype=torch.float)


class MolDataset(InMemoryDataset):
    """
    A PyTorch Geometric dataset for small molecules. For each molecule, the dataset contains the following:
        - x: Node features (atom type, degree, number of Hs, hybridization, aromaticity, presence in rings of size 3-6)
        - edge_index: Edge connectivity
        - edge_attr: Edge features (bond type)
        - pos: Node positions
        - y: Target value (activity)
        - name: Molecule name
    """

    url = 'https://github.com/vladislach/small-molecules/raw/main/structures.zip'

    def __init__(self, root, csv, transform=None):
        self.df = pd.read_csv(csv)
        super().__init__(root, transform)
        self.load(self.processed_paths[0])
    
    def raw_file_names(self):
        return [f"{id}.mol2" for id in self.df['id']]
    
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
    
    def process(self):
        data_list = []

        for path, row in zip(self.raw_paths, self.df.iterrows()):
            row = row[1]
            rdmol = Chem.MolFromMol2File(path)
            x = self.get_node_feats(rdmol)
            edge_index = self.get_edge_index(rdmol)
            edge_attr = self.get_edge_attr(rdmol, edge_index)
            pos = torch.tensor(rdmol.GetConformer().GetPositions(), dtype=torch.float)
            y = row['activity']
            name = row['name']
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos, name=name)
            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
    
    def get_node_feats(self, rdmol: Chem.Mol) -> torch.Tensor:
        """Generates a feature tensor for each atom (node) in an RDKit molecule object (rdmol)."""
        feats = []
        feats.append(
            one_hot(
                torch.tensor([x_map['atomic_num'].index(atom.GetAtomicNum()) for atom in rdmol.GetAtoms()]),
                num_classes=len(x_map['atomic_num']), dtype=torch.float
            )
        )
        feats.append(
            one_hot(
                torch.tensor([x_map['degree'].index(atom.GetTotalDegree()) for atom in rdmol.GetAtoms()]),
                num_classes=len(x_map['degree']), dtype=torch.float
            )
        )
        feats.append(
            one_hot(
                torch.tensor([x_map['num_hs'].index(atom.GetTotalNumHs()) for atom in rdmol.GetAtoms()]),
                num_classes=len(x_map['num_hs']), dtype=torch.float
            )
        )
        feats.append(
            one_hot(
                torch.tensor([x_map['hybridization'].index(str(atom.GetHybridization())) for atom in rdmol.GetAtoms()]),
                num_classes=len(x_map['hybridization']), dtype=torch.float
            )
        )
        feats.append(
            torch.tensor([atom.GetIsAromatic() for atom in rdmol.GetAtoms()], dtype=torch.float).view(-1, 1)
        )
        for i in range(3, 7):
            feats.append(
                torch.tensor([atom.IsInRingSize(i) for atom in rdmol.GetAtoms()], dtype=torch.float).view(-1, 1)
            )
        return torch.cat(feats, dim=-1)
    
    def get_edge_index(self, rdmol: Chem.Mol) -> torch.Tensor:
        """Generates an edge index tensor (edge connectivity) for an RDKit molecule object (mol)."""
        return torch.tensor(Chem.GetAdjacencyMatrix(rdmol), dtype=torch.long).to_sparse().indices()
    
    def get_edge_attr(self, rdmol: Chem.Mol, edge_index: torch.Tensor) -> torch.Tensor:
        """Generates a feature tensor for each bond (edge) in an RDKit molecule object (rdmol)."""
        edge_attr = []
        for i, j in edge_index.T:
            bond = rdmol.GetBondBetweenAtoms(i.item(), j.item())
            edge_attr.append(e_map['bond_type'].index(str(bond.GetBondType())))

        edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        return one_hot(edge_attr, num_classes=len(e_map['bond_type']), dtype=torch.float)
    