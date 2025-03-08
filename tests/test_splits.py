import numpy as np

from mgb.splits import murcko_scaffold, random_split, scaffold_overlap, scaffold_split

SMILES = [
    "c1ccccc1C", "c1ccccc1CC", "c1ccccc1O", "c1ccccc1N",
    "C1CCCCC1C", "C1CCCCC1O", "C1CCCCC1N",
    "c1ccncc1C", "c1ccncc1O",
    "CCO", "CCN", "CCC", "CCCl",
    "c1ccc2ccccc2c1C", "c1ccc2ccccc2c1O",
    "C1CC1C", "C1CC1O", "C1CCC1C", "C1CCC1O", "c1ccoc1C",
]


def test_scaffold_groups_share_scaffold():
    assert murcko_scaffold("c1ccccc1C") == murcko_scaffold("c1ccccc1O")
    assert murcko_scaffold("c1ccccc1C") != murcko_scaffold("C1CCCCC1C")


def test_partition_is_complete_and_disjoint():
    tr, va, te = scaffold_split(SMILES, seed=0)
    all_idx = np.concatenate([tr, va, te])
    assert sorted(all_idx) == list(range(len(SMILES)))


def test_no_scaffold_leakage_between_train_and_test():
    tr, va, te = scaffold_split(SMILES, seed=0)
    train_scaf = {murcko_scaffold(SMILES[i]) for i in tr}
    test_scaf = {murcko_scaffold(SMILES[i]) for i in te}
    # acyclic molecules share the empty scaffold; ignore that group
    assert not (train_scaf & test_scaf - {""})


def test_seeds_change_the_split():
    a = scaffold_split(SMILES, seed=0)[2]
    b = scaffold_split(SMILES, seed=1)[2]
    c = scaffold_split(SMILES, seed=2)[2]
    assert not (np.array_equal(a, b) and np.array_equal(b, c))


def test_random_split_sizes():
    tr, va, te = random_split(100, seed=0)
    assert (len(tr), len(va), len(te)) == (80, 10, 10)


def test_overlap_is_higher_for_random_than_scaffold():
    rng_tr, _, rng_te = random_split(len(SMILES), seed=0)
    sc_tr, _, sc_te = scaffold_split(SMILES, seed=0)
    assert scaffold_overlap(SMILES, sc_tr, sc_te) <= scaffold_overlap(SMILES, rng_tr, rng_te)
