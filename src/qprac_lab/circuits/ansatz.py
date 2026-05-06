from __future__ import annotations


def describe_hardware_efficient_ansatz(num_qubits: int, reps: int = 2):
    """Backend-neutral ansatz descriptor.

    Replace with Qiskit RealAmplitudes or EfficientSU2 in implementation phase.
    """
    return {
        "name": "hardware_efficient_ansatz",
        "num_qubits": num_qubits,
        "reps": reps,
    }
