import torch

from mgb.featurize import ATOM_DIM, BOND_DIM, collate, ecfp, mol_to_graph


def test_graph_shapes_for_ethanol():
    g = mol_to_graph("CCO")
    assert g.x.shape == (3, ATOM_DIM)
    assert g.edge_index.shape == (2, 4)  # 2 bonds, both directions
    assert g.edge_attr.shape == (4, BOND_DIM)


def test_single_atom_molecule_has_no_edges():
    g = mol_to_graph("C")
    assert g.x.shape[0] == 1
    assert g.edge_index.shape == (2, 0)


def test_invalid_smiles_returns_none():
    assert mol_to_graph("not a molecule") is None


def test_collate_offsets_edge_indices():
    a, b = mol_to_graph("CCO"), mol_to_graph("CC")
    batch = collate([a, b])
    assert batch.n_graphs == 2
    assert batch.x.shape[0] == 5
    assert batch.edge_index[:, a.edge_index.shape[1]:].min() >= 3
    assert torch.equal(batch.batch, torch.tensor([0, 0, 0, 1, 1]))


def test_ecfp_is_deterministic_and_nonzero():
    a, b = ecfp("c1ccccc1O"), ecfp("c1ccccc1O")
    assert a.shape == (2048,)
    assert (a == b).all() and a.sum() > 0
