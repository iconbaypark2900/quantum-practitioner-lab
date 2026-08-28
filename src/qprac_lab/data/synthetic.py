from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification


def make_binary_classification_dataset(
    n_samples: int = 120,
    n_features: int = 4,
    random_state: int = 42,
):
    """Create a compact dataset for QSVC/VQC tutorials.

    Algorithm context:
    - Classical synthetic data generation.
    - Used to benchmark quantum kernel and variational classifiers.
    """
    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    return x, y


def make_small_portfolio_dataset(n_assets: int = 6, random_state: int = 42):
    """Generate expected returns and covariance for portfolio optimization demos."""
    rng = np.random.default_rng(random_state)
    expected_returns = rng.uniform(0.02, 0.20, size=n_assets)
    random_matrix = rng.normal(size=(n_assets, n_assets))
    covariance = random_matrix.T @ random_matrix
    covariance = covariance / covariance.max()
    return expected_returns, covariance


def make_biomedical_pair_features(
    n_pairs: int = 120,
    embedding_dim: int = 4,
    random_state: int = 42,
):
    """Gaussian-blob stand-in for KG pair features -- an offline fallback only.

    Superseded by :func:`qprac_lab.data.hetionet.make_hetionet_link_prediction_dataset`,
    which builds real drug--disease features from Hetionet. Kept for environments
    without the downloaded data.

    Do not draw conclusions from it. ``make_classification`` blobs are close to
    the ideal geometry for an RBF kernel, so a quantum-versus-classical
    comparison run here measures the generator rather than the methods -- as the
    quantum-kernel tutorial found when the two datasets disagreed.
    """
    return make_binary_classification_dataset(
        n_samples=n_pairs,
        n_features=embedding_dim,
        random_state=random_state,
    )
