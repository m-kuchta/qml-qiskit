import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import (
    n_local,
)

# -------------------Feature Maps-------------------


def angle_fmap(
    feature_dimension: int, reps: int, insert_barriers: bool = True
) -> QuantumCircuit:
    """
    Encodes data using standard Angle Encoding with RY gates (one feature per qubit).

    Args:
        feature_dimension (int): Number of features (equals the number of qubits).
        reps (int): Number of times to repeat the encoding layer.
        insert_barriers (bool): Whether to insert barriers between repetitions.

    Returns:
        QuantumCircuit: The constructed feature map circuit.
    """
    fmap = QuantumCircuit(feature_dimension)
    x = ParameterVector("x", feature_dimension)
    for _ in range(reps):
        for i in range(feature_dimension):
            fmap.ry(x[i], i)
        if insert_barriers:
            fmap.barrier()

    return fmap


def dense_angle_fmap(
    num_qubits: int, reps: int, insert_barriers: bool = True
) -> QuantumCircuit:
    """
    Encodes data using Dense Angle Encoding with RY and RZ gates (two features per qubit).

    Args:
        num_qubits (int): Number of qubits .
        reps (int): Number of times to repeat the encoding layer.
        insert_barriers (bool): Whether to insert barriers between repetitions.

    Returns:
        QuantumCircuit: The constructed feature map circuit.
    """
    qc = QuantumCircuit(num_qubits)
    x = ParameterVector("x", num_qubits * 2)  # Two parameters per qubit
    for _ in range(reps):
        for i in range(int(num_qubits)):
            qc.ry(x[2 * i], i)
            qc.rz(x[2 * i + 1], i)
        if insert_barriers:
            qc.barrier()

    return qc


def phase_cry_fmap(
    num_qubits: int, reps: int, insert_barriers: bool = True
) -> QuantumCircuit:
    """
    Encodes data using Phase gates and Controlled-RY entanglement.

    Args:
        num_qubits (int): Number of qubits.
        reps (int): Number of times to repeat the encoding layer.
        insert_barriers (bool): Whether to insert barriers between repetitions.

    Returns:
        QuantumCircuit: The constructed feature map circuit.
    """
    qc = QuantumCircuit(num_qubits)
    x = ParameterVector("x", num_qubits * 2)  # Two parameters per qubit
    for _ in range(reps):
        for i in range(int(num_qubits)):
            qc.h(i)
            qc.p(x[i], i)

        if num_qubits > 1:
            for i in range(int(num_qubits)):
                target = (i + 1) % num_qubits
                qc.cry(x[num_qubits + i], i, target)

        if insert_barriers:
            qc.barrier()
    return qc


# -------------------Ansatze-------------------


def RY_RZ_CRX_ansatz(
    num_qubits: int,
    entanglement: str = "linear",
    reps: int = 1,
    insert_barriers: bool = True,
) -> QuantumCircuit:
    """
    Constructs an n-local ansatz using RY and RZ rotations
    with CRX entangling gates.

    Args:
        num_qubits (int): Number of qubits in the quantum circuit.
        entanglement (str): Entanglement topology (e.g., "linear", "circular").
        reps (int): Number of repetition layers for the ansatz.
        insert_barriers (bool): Whether to insert barriers between repetitions.

    Returns:
        QuantumCircuit: The constructed ansatz circuit.
    """
    qc = n_local(
        num_qubits=num_qubits,
        rotation_blocks=["ry", "rz"],
        entanglement_blocks=["crx"],
        entanglement=entanglement,
        reps=reps,
        insert_barriers=insert_barriers,
    )
    return qc


def RY_RZ_RXX_ansatz(
    num_qubits: int,
    entanglement: str = "linear",
    reps: int = 1,
    insert_barriers: bool = True,
) -> QuantumCircuit:
    """
    Constructs an n-local ansatz using RY and RZ rotations
    with RXX entangling gates.

    Args:
        num_qubits (int): Number of qubits in the quantum circuit.
        entanglement (str): Entanglement topology (e.g., "linear", "circular").
        reps (int): Number of repetition layers for the ansatz.
        insert_barriers (bool): Whether to insert barriers between repetitions.

    Returns:
        QuantumCircuit: The constructed ansatz circuit.
    """
    qc = n_local(
        num_qubits=num_qubits,
        rotation_blocks=["ry", "rz"],
        entanglement_blocks=["rxx"],
        entanglement=entanglement,
        reps=reps,
        insert_barriers=insert_barriers,
    )
    return qc


def overlapped_ansatz(
    num_qubits: int,
    entanglement: str = "linear",
    reps: int = 1,
    insert_barriers: bool = True,
) -> QuantumCircuit:
    """
    Constructs a custom ansatz using alternating single-qubit
    rotations and RXX entangling gates across specified pair topologies.

    Args:
        num_qubits (int): Number of qubits in the quantum circuit.
        entanglement (str): Entanglement topology, either "linear" or "circular".
        reps (int): Number of repetition layers for the ansatz.
        insert_barriers (bool): Whether to insert barriers between repetitions.

    Returns:
        QuantumCircuit: The constructed ansatz circuit.

    Raises:
        ValueError: If the entanglement type is not recognized.
    """
    qc = QuantumCircuit(num_qubits)

    # Determine pairs for Linear/Circular entanglement
    if entanglement == "linear":
        pairs = [(i, i + 1) for i in range(num_qubits - 1)]
    elif entanglement == "circular":  # circular
        pairs = [(i, (i + 1) % num_qubits) for i in range(num_qubits)]
    else:
        raise ValueError("Entanglement must be 'linear' or 'circular'")

    # 3 parameters per qubit per rep (RY, RXX, RZ)
    theta = ParameterVector(
        "θ",
        len(pairs) * 3 * reps
        if entanglement == "circular"
        else (num_qubits * 2 + len(pairs)) * reps,
    )

    # Parameter counting gets tricky here, better to use a simple counter:
    p_idx = 0

    for _ in range(reps):
        for _, (control, target) in enumerate(pairs):
            # 1. Rotate the control qubit
            qc.ry(theta[p_idx], control)
            p_idx += 1

            # 2. Immediately entangle with the next
            qc.rxx(theta[p_idx], control, target)
            p_idx += 1

            # 3. Apply a phase rotation to the control
            qc.rz(theta[p_idx], control)
            p_idx += 1

        if insert_barriers:
            qc.barrier()

    return qc


def double_entanglement_ansatz(
    num_qubits: int,
    entanglement: str = "linear",
    reps: int = 1,
    insert_barriers: bool = True,
) -> QuantumCircuit:
    """
    Constructs a ansatz featuring alternating RY rotations
    and a double-entanglement layer (parameterized RZZ followed by CX gates)
    with linear or circular topologies, concluded by a final RY rotation layer.

    Args:
        num_qubits (int): Number of qubits in the quantum circuit.
        entanglement (str): Entanglement topology, either "linear" or "circular".
        reps (int): Number of repetition layers for the ansatz.
        insert_barriers (bool): Whether to insert barriers between repetitions.

    Returns:
        QuantumCircuit: The constructed ansatz circuit.

    Raises:
        ValueError: If the entanglement type is not recognized.
    """

    qc = QuantumCircuit(num_qubits)

    # Determine pairs for Linear/Circular entanglement
    if entanglement == "circular":
        pairs = [(i, (i + 1) % num_qubits) for i in range(num_qubits)]
    elif entanglement == "linear":
        pairs = [(i, i + 1) for i in range(num_qubits - 1)]
    else:
        raise ValueError("Entanglement must be 'linear' or 'circular'")

    num_pairs = len(pairs)

    # Calculate theta count:
    # (RY_layer + RZZ_layer) * reps + Final_RY_layer
    total_params = (num_qubits + num_pairs) * reps + num_qubits
    theta = ParameterVector("θ", total_params)

    p_idx = 0

    for _ in range(reps):
        # Rotation Layer
        for i in range(num_qubits):
            qc.ry(theta[p_idx], i)
            p_idx += 1

        # Entanglement Layer
        for control, target in pairs:
            # Apply parameterized RZZ
            qc.rzz(theta[p_idx], control, target)
            p_idx += 1
            # Apply CX
            qc.cx(control, target)

        if insert_barriers:
            qc.barrier()

    # Final Rotation Layer
    for i in range(num_qubits):
        qc.ry(theta[p_idx], i)
        p_idx += 1

    return qc


def u_cu_ansatz(
    num_qubits: int,
    entanglement: str = "linear",
    reps: int = 1,
    insert_barriers: bool = True,
) -> QuantumCircuit:
    """
    Constructs a ansatz using parameterized general single-qubit
    unitary gates (U) followed by controlled-unitary gates (CU)
    with linear or circular entanglement topology.

    Args:
        num_qubits (int): Number of qubits in the quantum circuit.
        entanglement (str): Entanglement topology, either "linear" or "circular".
        reps (int): Number of repetition layers for the ansatz.
        insert_barriers (bool): Whether to insert barriers between repetitions.

    Returns:
        QuantumCircuit: The constructed ansatz circuit.

    Raises:
        ValueError: If the entanglement type is not recognized.
    """

    qc = QuantumCircuit(num_qubits)

    # Determine pairs for Linear/Circular entanglement
    if entanglement == "circular":
        pairs = [(i, (i + 1) % num_qubits) for i in range(num_qubits)]
    elif entanglement == "linear":
        pairs = [(i, i + 1) for i in range(num_qubits - 1)]
    else:
        raise ValueError("Entanglement must be 'linear' or 'circular'")

    num_pairs = len(pairs)
    params_per_gate = 3

    total_params = (num_qubits * params_per_gate + num_pairs * params_per_gate) * reps

    theta = ParameterVector("θ", total_params)
    p_idx = 0

    for r in range(reps):
        #  U Layer (Rotation on every qubit)
        for i in range(num_qubits):
            qc.u(theta[p_idx], theta[p_idx + 1], theta[p_idx + 2], i)
            p_idx += 3

        # CU Entanglement Layer ---
        for control, target in pairs:
            # Gamma = 0, using only theta, phi, lambda
            qc.cu(theta[p_idx], theta[p_idx + 1], theta[p_idx + 2], 0, control, target)
            p_idx += 3

        if insert_barriers:
            qc.barrier()

    return qc
