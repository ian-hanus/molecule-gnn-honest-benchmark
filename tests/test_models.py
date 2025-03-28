import pytest
import torch

from mgb.featurize import collate, mol_to_graph
from mgb.models import MoleculeGNN

SMILES = ["CCO", "c1ccccc1O", "CC(=O)N", "C"]


def _batch(smiles):
    return collate([mol_to_graph(s) for s in smiles])


@pytest.mark.parametrize("arch", ["gcn", "gin", "dmpnn"])
def test_output_shape_and_finite(arch):
    model = MoleculeGNN(arch, n_tasks=3).eval()
    out = model(_batch(SMILES))
    assert out.shape == (len(SMILES), 3)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("arch", ["gcn", "gin", "dmpnn"])
def test_prediction_is_independent_of_batch_composition(arch):
    torch.manual_seed(0)
    model = MoleculeGNN(arch, n_tasks=1).eval()
    alone = model(_batch(["c1ccccc1O"]))
    together = model(_batch(SMILES))[1:2]
    assert torch.allclose(alone, together, atol=1e-5)


@pytest.mark.parametrize("arch", ["gcn", "gin", "dmpnn"])
def test_equivalent_atom_orderings_give_same_output(arch):
    torch.manual_seed(0)
    model = MoleculeGNN(arch, n_tasks=1).eval()
    a = model(_batch(["OCC"]))
    b = model(_batch(["CCO"]))
    assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("arch", ["gcn", "gin", "dmpnn"])
def test_gradients_flow(arch):
    model = MoleculeGNN(arch, n_tasks=1)
    model(_batch(SMILES)).sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
