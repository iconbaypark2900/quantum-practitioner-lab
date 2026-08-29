"""The Hetionet link-prediction dataset.

Most of these assert the *construction* is honest rather than that the numbers
are good: no label leakage, balanced classes, and negatives sampled so that node
degree alone cannot solve the task.
"""

import numpy as np
import pytest

from qprac_lab.data.hetionet import (
    FEATURE_METAEDGES,
    FEATURE_NAMES,
    LEAKING_METAEDGES,
    TARGET_METAEDGE,
    degree_only_roc_auc,
    hetionet_available,
    make_hetionet_link_prediction_dataset,
)

needs_data = pytest.mark.skipif(
    not hetionet_available(),
    reason="Hetionet not cached; run `python scripts/download_data.py`",
)


def test_feature_edges_never_include_the_label_edge():
    """Structural no-leakage guarantee: no feature path touches CtD or CpD."""
    assert TARGET_METAEDGE in LEAKING_METAEDGES
    assert not set(FEATURE_METAEDGES) & set(LEAKING_METAEDGES)


def test_degree_features_are_ordered_last():
    """``embedding_dim=4`` must select biology features, not raw degrees."""
    assert FEATURE_NAMES[-2:] == ("compound_degree", "disease_degree")
    assert "degree" not in " ".join(FEATURE_NAMES[:4])


def test_degree_only_auc_helper_detects_a_shortcut():
    """A deliberately leaky degree feature must be flagged, or the check is useless."""
    rng = np.random.default_rng(0)
    labels = np.r_[np.ones(100), np.zeros(100)].astype(int)
    leaky = np.column_stack([labels * 5 + rng.normal(0, 0.1, 200), rng.normal(0, 1, 200)])
    assert degree_only_roc_auc(leaky, labels) > 0.95

    neutral = rng.normal(0, 1, size=(200, 2))
    assert 0.3 < degree_only_roc_auc(neutral, labels) < 0.7


@needs_data
def test_dataset_is_balanced_and_shaped_correctly():
    dataset = make_hetionet_link_prediction_dataset(n_pairs=120, with_diagnostics=False)
    assert len(dataset) == 120
    assert dataset.features.shape == (120, 4)
    assert dataset.labels.mean() == pytest.approx(0.5)
    assert set(np.unique(dataset.labels)) == {0, 1}
    assert dataset.feature_names == FEATURE_NAMES[:4]


@needs_data
def test_negative_sampling_removes_the_degree_shortcut():
    """The headline honesty claim, measured rather than asserted."""
    dataset = make_hetionet_link_prediction_dataset()
    assert dataset.metadata["degree_only_roc_auc"] < 0.60


@needs_data
def test_metadata_records_provenance():
    metadata = make_hetionet_link_prediction_dataset(n_pairs=60).metadata
    assert "Hetionet" in metadata["source"]
    assert metadata["license"].startswith("CC0")
    assert metadata["target_edge_type"].startswith(TARGET_METAEDGE)
    assert set(metadata["excluded_edge_types"]) == set(LEAKING_METAEDGES)
    assert metadata["degree_only_roc_auc_scope"] == "full dataset, before any subsampling"


@needs_data
def test_dataset_is_deterministic_for_a_seed():
    first = make_hetionet_link_prediction_dataset(n_pairs=80, seed=7, with_diagnostics=False)
    second = make_hetionet_link_prediction_dataset(n_pairs=80, seed=7, with_diagnostics=False)
    assert np.array_equal(first.features, second.features)
    assert np.array_equal(first.labels, second.labels)


@needs_data
def test_features_carry_more_signal_than_degree():
    """Biology features must beat the degree shortcut, or the task is not measuring biology."""
    dataset = make_hetionet_link_prediction_dataset(embedding_dim=4)
    biology_auc = degree_only_roc_auc(dataset.features, dataset.labels)
    assert biology_auc > dataset.metadata["degree_only_roc_auc"]


def test_embedding_dim_is_validated():
    with pytest.raises(ValueError):
        make_hetionet_link_prediction_dataset(embedding_dim=0, allow_download=False)
    with pytest.raises(ValueError):
        make_hetionet_link_prediction_dataset(embedding_dim=99, allow_download=False)


def test_missing_cache_without_download_raises_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        make_hetionet_link_prediction_dataset(cache_dir=tmp_path, allow_download=False)
