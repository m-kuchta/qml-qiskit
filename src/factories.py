import math
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    EfficientSU2,
    RealAmplitudes,
    StatePreparation,
    ZFeatureMap,
    ZZFeatureMap,
)
from qiskit.providers import BackendV2
from qiskit.quantum_info import SparsePauliOp

# from src.quantum_circuits import feature_map_custom_rot, create_custom_two_local
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import QiskitRuntimeService

# Import potrzebny do testowania szumów bez zużywania kredytów IBM
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

from src.quantum_circuits import (
    RY_RZ_CRX_ansatz,
    RY_RZ_RXX_ansatz,
    angle_fmap,
    dense_angle_fmap,
    double_entanglement_ansatz,
    overlapped_ansatz,
    phase_cry_fmap,
    u_cu_ansatz,
)


class CircuitFactory:
    """
    Factory class to construct quantum circuits for Variational Quantum Classifiers.
    """

    def __init__(
        self,
        num_features: int,
        fmap_name: str,
        fmap_reps: int,
        fmap_entanglement: str,
        ansatz_name: str,
        ansatz_reps: int,
        ansatz_entanglement: str,
    ) -> None:
        """
        Initialize the CircuitFactory with circuit parameters.

        Args:
            num_features (int): Number of features in the dataset.
            fmap_name (str): Name of the feature map architecture.
            fmap_reps (int): Number of repetitions for the feature map.
            fmap_entanglement (str): Entanglement topology for the feature map.
            ansatz_name (str): Name of the ansatz architecture.
            ansatz_reps (int): Number of repetitions for the ansatz.
            ansatz_entanglement (str): Entanglement topology for the ansatz.
        """
        # Set core parameters
        self.num_features: int = num_features
        self.fmap_name: str = fmap_name
        self.fmap_reps: int = fmap_reps
        self.fmap_entanglement: str = fmap_entanglement
        self.ansatz_name: str = ansatz_name
        self.ansatz_reps: int = ansatz_reps
        self.ansatz_entanglement: str = ansatz_entanglement

        # Calculate required qubits immediately upon initialization
        self.num_qubits: int = self.calculate_num_qubits()

    def calculate_num_qubits(self) -> int:
        """
        Calculate the required number of qubits based on the selected feature map.

        Returns:
            int: The calculated number of qubits.

        Raises:
            ValueError: If the feature map name is not recognized.
        """
        # Determine the number of qubits based on the feature dimension
        if self.fmap_name in ["Z", "ZZ", "Angle"]:
            return int(self.num_features)
        elif self.fmap_name in ["DenseAngle", "PhaseCRY"]:
            # Round up to accommodate odd numbers of features
            return math.ceil(self.num_features / 2.0)
        elif self.fmap_name in ["Amplitude"]:
            # Round up to the nearest power of 2
            return math.ceil(np.log2(self.num_features))
        else:
            raise ValueError("Feature map is not defined.")

    def build(
        self, data: np.ndarray | None = None
    ) -> tuple[QuantumCircuit | list[QuantumCircuit], bool]:
        """
        Construct the final circuit.

        Generates either a single parameterized blueprint circuit or a list of
        circuits with hardcoded features if data is provided (used mostly for encodings that are dependent on the input data).

        Args:
            data (Optional[np.ndarray]): The dataset with features to bind. Defaults to None.

        Returns:
            tuple[QuantumCircuit | list[QuantumCircuit], bool]: A tuple containing:
                - The generated circuit(s).
                - A boolean flag indicating if the output is a single blueprint (True) or a list of bound circuits (False).

        Raises:
            ValueError: If Amplitude encoding is selected but no data is provided, or if an architecture is not implemented.
        """
        # Initialize feature map
        fmap: QuantumCircuit | None = None
        if self.fmap_name == "Z":
            fmap = ZFeatureMap(
                feature_dimension=self.num_qubits,
                reps=self.fmap_reps,
                insert_barriers=True,
            )
        elif self.fmap_name == "ZZ":
            fmap = ZZFeatureMap(
                feature_dimension=self.num_qubits,
                reps=self.fmap_reps,
                entanglement=self.fmap_entanglement,
                insert_barriers=True,
            )
        elif self.fmap_name == "Angle":
            fmap = angle_fmap(
                feature_dimension=self.num_qubits,
                reps=self.fmap_reps,
                insert_barriers=True,
            )
        elif self.fmap_name == "DenseAngle":
            fmap = dense_angle_fmap(
                num_qubits=self.num_qubits,
                reps=self.fmap_reps,
                insert_barriers=True,
            )
        elif self.fmap_name == "PhaseCRY":
            fmap = phase_cry_fmap(
                num_qubits=self.num_qubits,
                reps=self.fmap_reps,
                insert_barriers=True,
            )
        elif self.fmap_name == "Amplitude":
            pass
        else:
            raise ValueError("Feature map not implemented yet!")

        # Initialize ansatz
        ansatz: QuantumCircuit | None = None
        if self.ansatz_name == "RealAmplitudes":
            ansatz = RealAmplitudes(
                num_qubits=self.num_qubits,
                entanglement=self.ansatz_entanglement,
                reps=self.ansatz_reps,
                insert_barriers=True,
            )
        elif self.ansatz_name == "EfficientSU2":
            ansatz = EfficientSU2(
                num_qubits=self.num_qubits,
                entanglement=self.ansatz_entanglement,
                reps=self.ansatz_reps,
                insert_barriers=True,
            )
        elif self.ansatz_name == "RY_RZ_CRX":
            ansatz = RY_RZ_CRX_ansatz(
                num_qubits=self.num_qubits,
                entanglement=self.ansatz_entanglement,
                reps=self.ansatz_reps,
                insert_barriers=True,
            )
        elif self.ansatz_name == "RY_RZ_RXX":
            ansatz = RY_RZ_RXX_ansatz(
                num_qubits=self.num_qubits,
                entanglement=self.ansatz_entanglement,
                reps=self.ansatz_reps,
                insert_barriers=True,
            )
        elif self.ansatz_name == "Overlapped":
            ansatz = overlapped_ansatz(
                num_qubits=self.num_qubits,
                entanglement=self.ansatz_entanglement,
                reps=self.ansatz_reps,
                insert_barriers=True,
            )
        elif self.ansatz_name == "DoubleEnt":
            ansatz = double_entanglement_ansatz(
                num_qubits=self.num_qubits,
                entanglement=self.ansatz_entanglement,
                reps=self.ansatz_reps,
                insert_barriers=True,
            )
        elif self.ansatz_name == "U_CU":
            ansatz = u_cu_ansatz(
                num_qubits=self.num_qubits,
                entanglement=self.ansatz_entanglement,
                reps=self.ansatz_reps,
                insert_barriers=True,
            )
        else:
            raise ValueError("Ansatz not implemented yet!")

        # Store the exact number of trainable weights (thetas) required by the ansatz
        self.num_weights: int = ansatz.num_parameters

        # Compose the final blueprint circuit
        if data is None:
            # Blueprint mode
            if self.fmap_name == "Amplitude":
                raise ValueError(
                    "Amplitude encoding requires actual data. Pass dataset to build() function."
                )

            assert fmap is not None
            assert ansatz is not None

            # Compose and return the blueprint
            blueprint = QuantumCircuit(self.num_qubits)
            blueprint.compose(fmap, inplace=True)
            blueprint.compose(ansatz, inplace=True)

            # Return the circuit and a boolean flag
            return blueprint, True

        else:
            # Bound circuits mode
            bound_circuits = []

            # Iterate over each row in the dataset
            for sample in data:
                final_qc = QuantumCircuit(self.num_qubits)

                if self.fmap_name == "Amplitude":
                    norm = np.linalg.norm(sample)
                    normalized_sample = sample / norm if norm != 0 else sample
                    sp = StatePreparation(normalized_sample)
                    final_qc.compose(sp, inplace=True)
                else:
                    assert fmap is not None
                    # Bind features to blueprint (mainly for testing)
                    bound_fmap = fmap.assign_parameters(sample)
                    assert bound_fmap is not None
                    final_qc.compose(bound_fmap, inplace=True)

                # Append the ansatz
                final_qc.compose(ansatz, inplace=True)
                bound_circuits.append(final_qc)

            # Return the list of circuits and a boolean flag
            return bound_circuits, False


def get_observable(
    observable_name: str, num_qubits: int, **kwargs: Any
) -> SparsePauliOp:
    """
    Generate the specified Pauli observable for the quantum circuit.

    Args:
        observable_name (str): The name of the observable.
        num_qubits (int): The number of qubits in the circuit.

    Returns:
        SparsePauliOp: The generated observable.

    Raises:
        ValueError: If the observable name is not recognized.
    """

    if observable_name == "GLOBAL_Z":
        # "ZZZ...Z" for all qubits
        return SparsePauliOp("Z" * num_qubits)

    elif observable_name in ["SUM_OF_LOCAL_Z"]:
        # "Z" on each qubit, summed together
        pauli_terms: list[tuple[str, float]] = [
            (f"{'I' * i}Z{'I' * (num_qubits - i - 1)}", 1.0) for i in range(num_qubits)
        ]
        return SparsePauliOp.from_list(pauli_terms)

    elif observable_name == "LOCAL_Z":
        # "Z" on a specific qubit, "I" on others
        position: int = kwargs.get("position", 0)

        # Validate that the position is within the valid qubit range
        if not (0 <= position < num_qubits):
            raise ValueError(
                f"Position {position} is out of bounds for {num_qubits} qubits."
            )

        pauli_str = f"{'I' * position}Z{'I' * (num_qubits - position - 1)}"
        pauli_terms = [(pauli_str, 1.0)]
        return SparsePauliOp.from_list(pauli_terms)

    else:
        raise ValueError(f"Observable '{observable_name}' is not implemented.")


def get_backend(
    config: dict,
) -> BackendV2:
    """
    Initializes and returns a Qiskit backend based on the provided configuration.

    The backend type is determined by `config["backend"]["type"]`, which can be:
    - "LOCAL": Returns a standard, ideal AerSimulator.
    - "FAKE": Returns a noisy fake backend mimicking real hardware.
    - "REAL": Connects to actual IBM Quantum hardware via QiskitRuntimeService.

    Args:
        config (dict): Configuration dictionary containing backend settings.

    Returns:
        BackendV2: The initialized Qiskit backend instance.

    Raises:
        ValueError: If an unknown backend type is provided.
    """
    backend_type = config.get("backend", {}).get("type", "LOCAL").upper()

    if backend_type == "LOCAL":
        print(">>> Initialize: Local backend (AerSimulator)")
        return AerSimulator()

    elif backend_type == "FAKE":
        backend_name = config["backend"].get("name", "fake_guadelupe")
        print(f">>> Initialize: Fake backend (AerSimulator + config) ({backend_name})")

        provider = FakeProviderForBackendV2()
        return provider.backend(backend_name)

    elif backend_type == "REAL":
        backend_name = config["backend"]["name"]
        print(f">>> Initialize: Real quantum hardware ({backend_name})")
        service = QiskitRuntimeService()
        return service.backend(backend_name)

    else:
        raise ValueError(f"Unknown backend type in config: {backend_type}")


def get_estimator(backend, config: dict, seed: int):
    """
    Configures and initializes a Qiskit Primitives Estimator based on the backend type.
    Sets up the execution options such as the number of shots and random seeds
    for reproducibility.

    Args:
        backend (BackendV2): The Qiskit backend instance to run the Estimator on.
        config (dict): Configuration dictionary containing estimator settings
            (e.g., 'estimator_shots').
        seed (int): The random seed for reproducibility (applied to simulators).

    Returns:
        Estimator: The configured Qiskit Estimator instance ready for execution.

    Raises:
        ValueError: If an unknown backend type is provided in the config.
    """
    backend_type = config.get("backend", {}).get("type", "LOCAL").upper()

    if backend_type in ["LOCAL", "FAKE"]:
        options = {
            "default_shots": config.get("estimator_shots", 1024),
            "seed_estimator": seed,
            "simulator": {"seed_simulator": seed},
        }
    elif backend_type == "REAL":
        options = {"default_shots": config.get("estimator_shots", 1024)}
    else:
        raise ValueError(f"Unknown backend type in config: {backend_type}")

    # Initialize the estimator
    estimator = Estimator(
        mode=backend,
        options=options,
    )

    return estimator
