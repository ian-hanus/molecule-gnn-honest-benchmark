import numpy as np

from mgb.metrics import ece, gaussian_nll, interval_coverage, mae, multitask_auc, rmse


def test_rmse_mae_known_values():
    y, p = np.array([0.0, 0.0]), np.array([3.0, 4.0])
    assert np.isclose(rmse(y, p), np.sqrt(12.5))
    assert np.isclose(mae(y, p), 3.5)


def test_auc_ignores_nan_and_single_class_tasks():
    y = np.array([[0, 1], [1, 1], [np.nan, 1], [0, 1]], dtype=float)
    p = np.array([[0.1, 0.5], [0.9, 0.5], [0.5, 0.5], [0.2, 0.5]])
    assert np.isclose(multitask_auc(y, p), 1.0)  # task 2 has one class -> skipped


def test_perfectly_calibrated_has_zero_ece():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    p = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    assert np.isclose(ece(y, p), 0.0)


def test_overconfident_predictions_have_high_ece():
    y = np.array([0, 1, 0, 1], dtype=float)
    p = np.array([0.99, 0.99, 0.99, 0.99])
    assert ece(y, p) > 0.4


def test_interval_coverage_and_nll():
    y = np.array([0.0, 0.0, 10.0])
    mu = np.zeros(3)
    assert np.isclose(interval_coverage(y, mu, np.ones(3)), 2 / 3)
    assert gaussian_nll(y, mu, np.full(3, 5.0)) < gaussian_nll(y, mu, np.full(3, 0.1))
