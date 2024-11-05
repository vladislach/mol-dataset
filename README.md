# Molecular Dataset in PyTorch Geometric

This repository contains a collection of code snippets and methods to create a custom A PyTorch Geometric dataset for small molecules. For each molecule, the dataset contains the following:

- x: Node features (atom type, degree, number of Hs, hybridization, aromaticity, presence in rings of size 3-6) obtained from RDKit
- edge_index: Edge connectivity
- edge_attr: Edge features (bond type)
- pos: Node positions
- y: Target value (activity)
- name: Molecule name

The fucntions and classes can be found in the `dataset.py` file. The `example.ipynb` notebook shows how to use them on a toy dataset.
