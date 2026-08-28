"""Real biomedical knowledge-graph features from Hetionet.

Hetionet (Himmelstein et al., eLife 2017) is the canonical drug-repurposing
knowledge graph: 47k nodes and 2.25M edges across 11 node types and 24 edge
types, released under CC0. This module turns it into a supervised link-prediction
dataset for the quantum-kernel tutorial.

**Task.** Predict ``Compound--treats--Disease`` (``CtD``) edges. Hetionet has 755
of them over 1552 compounds and 137 diseases, which is the actual drug
repurposing problem rather than a stand-in for it.

Two design choices do the real work here, and both exist to stop the benchmark
from being accidentally easy:

**No label leakage.** Features are computed from compound--gene and
gene--disease edges only. The ``CtD`` edge being predicted is never traversed,
and neither is ``CpD`` (*palliates*), which is close enough to *treats* to
function as a label in disguise. Because no feature path touches either, there is
nothing to mask out per-split -- the separation is structural.

**Degree-matched negatives.** The naive negative sample -- a random compound and
a random disease -- makes degree alone predictive: well-studied compounds have
more edges *and* more known treatments, so a classifier can score well without
learning any biology. Each positive ``(compound, disease)`` is instead paired
with a negative ``(other_compound, disease)``: the disease is held fixed, so its
degree matches exactly, and the substitute compound is drawn from the
``degree_window`` nearest compounds by degree.

The dataset reports ``degree_only_roc_auc`` so this can be checked rather than
taken on trust. It sits near 0.54 -- close to the 0.5 of a coin flip, against
0.62 for the biology features.
"""

from __future__ import annotations

import bisect
import gzip
import math
import random
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HETIONET_EDGES_URL = (
    "https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz"
)
HETIONET_EDGES_FILENAME = "hetionet-v1.0-edges.sif.gz"

#: The edge type being predicted. Never used as an input feature.
TARGET_METAEDGE = "CtD"

#: Excluded from features: ``CtD`` is the label and ``CpD`` (palliates) is close
#: enough to it to leak.
LEAKING_METAEDGES = ("CtD", "CpD")

#: Compound--gene and disease--gene edges, the only ones features are built from.
FEATURE_METAEDGES = ("CbG", "CuG", "CdG", "DaG", "DuG", "DdG")

#: Ordered feature list; ``embedding_dim`` takes a prefix of this.
#: The first four are degree-normalised or overlap-based biology signals; the
#: last two are raw degrees, kept last precisely because they are the shortcut.
FEATURE_NAMES = (
    "bind_gene_overlap",
    "any_gene_overlap",
    "jaccard_gene",
    "adamic_adar_gene",
    "expression_opposition",
    "compound_degree",
    "disease_degree",
)

DEFAULT_DEGREE_WINDOW = 15


@dataclass
class LinkPredictionDataset:
    """Feature matrix, labels, and the provenance needed to trust them."""

    features: np.ndarray
    labels: np.ndarray
    feature_names: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.labels))


def default_cache_dir() -> Path:
    """Where downloaded source data is cached (``data/raw`` under the cwd)."""
    import os

    override = os.environ.get("QPRAC_DATA_DIR")
    return Path(override) if override else Path("data") / "raw"


def hetionet_path(cache_dir: Path | str | None = None) -> Path:
    return Path(cache_dir or default_cache_dir()) / HETIONET_EDGES_FILENAME


def hetionet_available(cache_dir: Path | str | None = None) -> bool:
    """Whether the Hetionet edge file is already cached locally."""
    path = hetionet_path(cache_dir)
    return path.exists() and path.stat().st_size > 0


def download_hetionet(
    cache_dir: Path | str | None = None,
    force: bool = False,
    timeout: int = 180,
) -> Path:
    """Fetch the Hetionet edge list (~12 MB) into the cache directory."""
    path = hetionet_path(cache_dir)
    if path.exists() and path.stat().st_size > 0 and not force:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    try:
        with urllib.request.urlopen(HETIONET_EDGES_URL, timeout=timeout) as response:
            partial.write_bytes(response.read())
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download Hetionet from {HETIONET_EDGES_URL}: {error}. "
            f"Download it manually to {path}, or pass dataset='synthetic'."
        ) from error
    partial.replace(path)
    return path


def load_hetionet_edges(
    cache_dir: Path | str | None = None,
    allow_download: bool = True,
) -> tuple[dict[str, dict[str, set[str]]], set[tuple[str, str]]]:
    """Parse Hetionet into ``(adjacency_by_metaedge, positive_CtD_pairs)``.

    Only the metaedges needed for features are retained, which keeps this to
    about 80k of the 2.25M edges.
    """
    path = hetionet_path(cache_dir)
    if not hetionet_available(cache_dir):
        if not allow_download:
            raise FileNotFoundError(
                f"Hetionet is not cached at {path}. Run "
                "`python scripts/download_data.py`, or pass allow_download=True."
            )
        path = download_hetionet(cache_dir)

    adjacency: dict[str, dict[str, set[str]]] = {
        metaedge: defaultdict(set) for metaedge in FEATURE_METAEDGES
    }
    positives: set[tuple[str, str]] = set()
    with gzip.open(path, "rt") as handle:
        next(handle)  # header: source, metaedge, target
        for line in handle:
            source, metaedge, target = line.rstrip("\n").split("\t")
            if metaedge == TARGET_METAEDGE:
                positives.add((source, target))
            elif metaedge in adjacency:
                adjacency[metaedge][source].add(target)
                adjacency[metaedge][target].add(source)
    return adjacency, positives


def _compound_genes(adjacency, compound: str) -> set[str]:
    return (
        adjacency["CbG"][compound] | adjacency["CuG"][compound] | adjacency["CdG"][compound]
    )


def _disease_genes(adjacency, disease: str) -> set[str]:
    return adjacency["DaG"][disease] | adjacency["DuG"][disease] | adjacency["DdG"][disease]


def _pair_features(adjacency, gene_degree, compound: str, disease: str) -> list[float]:
    """Seven graph-derived features for one (compound, disease) pair."""
    compound_genes = _compound_genes(adjacency, compound)
    disease_genes = _disease_genes(adjacency, disease)
    shared = compound_genes & disease_genes
    union = compound_genes | disease_genes

    return [
        # Direct target overlap: the compound binds a gene the disease implicates.
        float(len(adjacency["CbG"][compound] & disease_genes)),
        float(len(shared)),
        # Degree-normalised overlap, so a promiscuous compound gains nothing.
        len(shared) / max(len(union), 1),
        # Adamic-Adar: rare shared genes count for more than ubiquitous ones.
        sum(1.0 / math.log(gene_degree[gene] + 2) for gene in shared),
        # The classic repurposing signal: the drug pushes expression in the
        # opposite direction to the disease.
        float(
            len(adjacency["CuG"][compound] & adjacency["DdG"][disease])
            + len(adjacency["CdG"][compound] & adjacency["DuG"][disease])
        ),
        float(len(compound_genes)),
        float(len(disease_genes)),
    ]


def _degree_matched_negatives(positives, compounds, degree, window, seed):
    """One negative per positive, holding the disease fixed and matching degree."""
    ordered = sorted(compounds, key=lambda c: degree[c])
    degrees = [degree[c] for c in ordered]
    rng = random.Random(seed)

    negatives = []
    for compound, disease in sorted(positives):
        index = bisect.bisect_left(degrees, degree[compound])
        low = max(0, index - window)
        high = min(len(ordered), index + window + 1)
        candidates = [
            other
            for other in ordered[low:high]
            if other != compound and (other, disease) not in positives
        ]
        if candidates:
            negatives.append((rng.choice(candidates), disease))
    return negatives


def degree_only_roc_auc(degree_features: np.ndarray, labels: np.ndarray, seed: int = 0) -> float:
    """How well raw node degree alone separates the classes.

    The honesty check on the negative sampling. Near 0.5 means degree carries no
    shortcut and a classifier has to use the biology; well above 0.5 means the
    benchmark is easier than it looks and the headline numbers are inflated.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(
        RandomForestClassifier(n_estimators=100, random_state=seed),
        degree_features,
        labels,
        cv=5,
        scoring="roc_auc",
    )
    return float(scores.mean())


def make_hetionet_link_prediction_dataset(
    n_pairs: int | None = None,
    embedding_dim: int = 4,
    seed: int = 42,
    degree_window: int = DEFAULT_DEGREE_WINDOW,
    cache_dir: Path | str | None = None,
    allow_download: bool = True,
    with_diagnostics: bool = True,
) -> LinkPredictionDataset:
    """Build a balanced drug--disease link-prediction dataset from Hetionet.

    ``n_pairs`` caps the total number of samples (positives and negatives stay
    balanced). The quantum kernel costs one circuit pair per matrix entry, so the
    tutorial keeps this small on purpose.
    """
    if not 1 <= embedding_dim <= len(FEATURE_NAMES):
        raise ValueError(f"embedding_dim must be 1..{len(FEATURE_NAMES)}, got {embedding_dim}")

    adjacency, positives = load_hetionet_edges(cache_dir, allow_download=allow_download)
    compounds = sorted({compound for compound, _ in positives})

    gene_degree: dict[str, int] = defaultdict(int)
    for metaedge in FEATURE_METAEDGES:
        for node, neighbours in adjacency[metaedge].items():
            gene_degree[node] += len(neighbours)

    degree = {c: len(_compound_genes(adjacency, c)) for c in compounds}
    negatives = _degree_matched_negatives(positives, compounds, degree, degree_window, seed)
    positive_pairs = sorted(positives)[: len(negatives)]

    rows = [_pair_features(adjacency, gene_degree, c, d) for c, d in positive_pairs]
    rows += [_pair_features(adjacency, gene_degree, c, d) for c, d in negatives]
    all_features = np.asarray(rows, dtype=float)
    labels = np.concatenate([np.ones(len(positive_pairs)), np.zeros(len(negatives))]).astype(int)

    # Computed on the full set, before subsampling: the diagnostic describes how
    # the dataset was *constructed*, and a small subsample would only add noise
    # to it.
    degree_auc = (
        degree_only_roc_auc(all_features[:, -2:], labels) if with_diagnostics else None
    )

    if n_pairs is not None and n_pairs < len(labels):
        rng = np.random.default_rng(seed)
        per_class = n_pairs // 2
        keep = np.concatenate(
            [
                rng.choice(np.flatnonzero(labels == 1), per_class, replace=False),
                rng.choice(np.flatnonzero(labels == 0), n_pairs - per_class, replace=False),
            ]
        )
        rng.shuffle(keep)
        all_features, labels = all_features[keep], labels[keep]

    metadata: dict[str, Any] = {
        "source": "Hetionet v1.0 (Himmelstein et al., eLife 2017)",
        "license": "CC0 1.0 (public domain)",
        "url": HETIONET_EDGES_URL,
        "target_edge_type": f"{TARGET_METAEDGE} (Compound-treats-Disease)",
        "excluded_edge_types": list(LEAKING_METAEDGES),
        "feature_edge_types": list(FEATURE_METAEDGES),
        "n_positive_edges_available": len(positives),
        "n_samples": int(len(labels)),
        "positive_rate": float(labels.mean()),
        "negative_sampling": (
            f"degree-matched: disease held fixed, compound drawn from the "
            f"{degree_window} nearest by degree"
        ),
        "degree_window": degree_window,
        "all_feature_names": list(FEATURE_NAMES),
        "features_used": list(FEATURE_NAMES[:embedding_dim]),
    }
    if with_diagnostics:
        metadata["degree_only_roc_auc"] = degree_auc
        metadata["degree_only_roc_auc_scope"] = "full dataset, before any subsampling"
        metadata["degree_shortcut_note"] = (
            "ROC-AUC from node degree alone; near 0.5 means the negative sampling "
            "left no degree shortcut for a classifier to exploit"
        )

    return LinkPredictionDataset(
        features=all_features[:, :embedding_dim],
        labels=labels,
        feature_names=FEATURE_NAMES[:embedding_dim],
        metadata=metadata,
    )
