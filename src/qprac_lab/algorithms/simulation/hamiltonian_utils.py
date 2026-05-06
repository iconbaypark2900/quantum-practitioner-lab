from __future__ import annotations


def describe_pauli_hamiltonian():
    """Return a small Hamiltonian description for tutorials."""
    return {
        "terms": [
            {"coefficient": -1.052373245772859, "pauli": "II"},
            {"coefficient": 0.39793742484318045, "pauli": "ZI"},
            {"coefficient": -0.39793742484318045, "pauli": "IZ"},
            {"coefficient": -0.01128010425623538, "pauli": "ZZ"},
            {"coefficient": 0.18093119978423156, "pauli": "XX"},
        ],
        "note": "Small H2-style Hamiltonian placeholder.",
    }
