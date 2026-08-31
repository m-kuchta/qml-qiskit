import gc
import time
from typing import Any

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import Bounds, minimize


def train_vqc(
    circuits: QuantumCircuit | list[QuantumCircuit],
    is_blueprint: bool,
    X_train: np.ndarray,
    y_train: np.ndarray,
    observable: SparsePauliOp,
    num_weights: int,
    initial_params: np.ndarray,
    config: dict,
    sample_weights: np.ndarray,
    estimator: Any,
    bounds: Bounds,
) -> tuple[np.ndarray, list[float], list[float]]:
    """
    Executes the main training loop for a Variational Quantum Classifier (VQC).

    This function uses SciPy's minimize() function to adjust the weights of the
    quantum circuit (ansatz) by minimizing the specified loss function.
    It uses EstimatorV2 from qiskit.ibm.runtime to evaluate the QNN using either a blueprint circuit or pre-bound data circuits.

    Args:
        circuits (QuantumCircuit | list[QuantumCircuit]): The parameterized quantum
            circuit (blueprint) or a list of circuits with features already bound.
        is_blueprint (bool): Flag indicating if `circuits` is a single template
            circuit (True) or a list of data-bound circuits (False).
        X_train (np.ndarray): Training data features.
        y_train (np.ndarray): Training data labels.
        observable (SparsePauliOp): The quantum observable to measure.
        num_weights (int): Total number of trainable parameters (theta) in the ansatz.
        config (dict): Configuration .yaml file containing optimizer settings,
            loss type, and theta ranges.
        sample_weights (np.ndarray): Array of weights for each training sample to
            handle class imbalance.
        num_qubits (int): Total number of qubits in the circuit.
        estimator (Any): The Qiskit Estimator V2 instance used to evaluate circuits.
        bounds (Bounds): SciPy Bounds object defining the parameter optimization space.

    Returns:
        tuple[np.ndarray, list[float], list[float]]: A tuple containing:
            - Optimized parameters array (result.x).
            - List of loss values recorded at each iteration.
            - List of accuracy values recorded at each iteration.
    """

    loss_history: list[float] = []
    accuracy_history: list[float] = []
    num_samples = len(X_train)
    loss_type = config.get("loss_type", "mse")

    iteration_counter = 0
    start_opt_time = time.time()

    print(f">>> Starting optimization loop (Loss: {loss_type.upper()})...")

    # CLOSURE: Cost function for the SciPy optimizer.
    # As a closure, it captures and accesses all variables defined in the
    # outer scope (e.g., X_train, y_train, circuits, observable).
    def evaluate_qnn(params: np.ndarray) -> float:

        nonlocal iteration_counter
        iteration_counter += 1

        # 1. Prepare PUBs (Primitive Unified Blocs) for the Estimator
        if is_blueprint:
            # BROADCASTING MODE for blueprints
            assert isinstance(circuits, QuantumCircuit)
            all_params = list(circuits.parameters)
            # Create index lists based on a naming convention:
            # - Feature Map parameters typically contain the letter 'x').
            # - All other parameters are assumed to be Ansatz weights 'θ'.
            feature_indices = [
                i for i, p in enumerate(all_params) if "x" in p.name.lower()
            ]
            weight_indices = [
                i for i, p in enumerate(all_params) if i not in feature_indices
            ]

            assert len(feature_indices) == X_train.shape[1], (
                "Number of 'x' parameters doesn't match X_train dimension!"
            )
            assert len(weight_indices) == num_weights, (
                "Number of weights doesn't match the number of ansatz parameters!"
            )

            # Initialize empty array for all parameters.
            # Shape: (num_samples, num_features + num_weights)
            param_array = np.zeros((num_samples, circuits.num_parameters))
            for i in range(num_samples):
                # Concatenate features and weights for a single row.
                # X_train[i] shape: (num_features,)
                # params shape:     (num_weights,)
                # Resulting shape:  (num_features + num_weights,)
                param_array[i, feature_indices] = X_train[i]
                param_array[i, weight_indices] = params

            # Create a SINGLE PUB.
            # Inputs:  (QuantumCircuit, SparsePauliOp, ndarray)
            # PUB structure: 1 circuit, 1 observable (can be multiple), param_array (num_samples x total_params)
            # Output expectation values (EVs) matrix shape (after job runs): (num_samples x num_observables)
            pubs = [(circuits, observable, param_array)]
        else:
            # BOUND CIRCUITS MODE for hardcoded data circuits

            # Create MULTIPLE PUBs (one for each circuit).
            # Inputs per PUB: (QuantumCircuit, SparsePauliOp, ndarray)
            # params shape for each PUB: (num_weights,)
            # pubs length: num_samples
            # Output EVs matrix shape (after job runs): list of `num_samples` single values
            pubs = [(circ, observable, params) for circ in circuits]

        # 2. Run the Estimator job to compute EVs
        job = estimator.run(pubs)
        result = job.result()

        # 3. Extract Expectation Values EVs
        if is_blueprint:
            evs = result[0].data.evs
        else:
            evs = np.array([res.data.evs for res in result])

        # Scale EVs if the observable has multiple Pauli terms
        num_terms = len(observable.paulis)
        print(f"Number of Pauli terms in observable: {num_terms}")
        if num_terms > 1:
            evs = evs / num_terms

        # 5. Calculate loss, accuracy, and predictions based on the specified loss type

        # Measuring in the Z-basis can yield either |0> or |1> states,
        # which correspond to expectation values of +1 and -1, respectively.
        # Mapping is as follows: Class 0 to |0> (+1) and Class 1 to |1> (-1).
        if loss_type == "mse":
            labels = np.where(y_train == 0, 1, -1)

            # If EV >= 0.0, the state is closer to |0>, meaning Class 0.
            # If EV < 0.0, the state is closer to |1>, meaning Class 1.
            predictions = np.where(evs >= 0.0, 0, 1)
            loss = np.average((evs - labels) ** 2, weights=sample_weights)

        elif loss_type == "cross_entropy":
            # Compute the probability of the state being |1> (Class 1) based on
            # the Bloch sphere mapping: EV = -1 -> Prob = 1.0, EV = +1 -> Prob = 0.0.
            probs = np.clip((1.0 - evs) / 2.0, 1e-15, 1 - 1e-15)
            labels = y_train

            # Predict Class 1 if the probability is >= 0.5, otherwise Class 0.
            predictions = np.where(probs >= 0.5, 1, 0)
            loss = -np.average(
                labels * np.log(probs) + (1 - labels) * np.log(1 - probs),
                weights=sample_weights,
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # 6. Calculate accuracy and log the results
        accuracy = np.mean(predictions == y_train)
        accuracy_history.append(accuracy)
        loss_history.append(loss)

        elapsed = time.time() - start_opt_time
        print(
            f"Iter: {iteration_counter:03d} | Accuracy: {accuracy * 100:.2f}% | Loss: {loss:.4f} | Mean EV: {np.mean(evs):.3f}| Time Elapsed: {elapsed:.1f}s"
        )

        del pubs, job, result, evs, predictions, labels
        gc.collect()

        return float(loss)

    # =====================================================
    # Run the SciPy optimizer to minimize the loss function
    # =====================================================
    result = minimize(
        fun=evaluate_qnn,
        x0=initial_params,
        method=config["optimizer"]["name"],  # e.g. 'COBYLA'
        tol=config["optimizer"]["tol"],
        options={
            "rhobeg": config["optimizer"]["rho_beg"],
            "maxiter": config["optimizer"]["num_epochs"],
            "disp": True,
        },
        bounds=bounds,
    )

    print(f">>> Optimization finished! Final Loss: {result.fun:.4f}")

    return result.x, loss_history, accuracy_history


def predict_vqc(
    circuits: QuantumCircuit | list[QuantumCircuit],
    is_blueprint: bool,
    X: np.ndarray,
    params: np.ndarray,
    observable: SparsePauliOp,
    num_weights: int,
    config: dict,
    estimator: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates predictions and expectation values for a given dataset and trained weights."""
    num_samples = len(X)
    loss_type = config.get("loss_type", "mse")

    # 1. Prepare PUBs
    if is_blueprint:
        assert isinstance(circuits, QuantumCircuit)
        all_params = list(circuits.parameters)
        feature_indices = [i for i, p in enumerate(all_params) if "x" in p.name.lower()]
        weight_indices = [
            i for i, p in enumerate(all_params) if i not in feature_indices
        ]

        assert len(feature_indices) == X.shape[1], (
            "Number of 'x' parameters doesn't match X dimension!"
        )
        assert len(weight_indices) == num_weights, (
            "Number of weights doesn't match the number of ansatz parameters!"
        )
        param_array = np.zeros((num_samples, circuits.num_parameters))
        for i in range(num_samples):
            param_array[i, feature_indices] = X[i]
            param_array[i, weight_indices] = params
        pubs = [(circuits, observable, param_array)]
    else:
        pubs = [(circ, observable, params) for circ in circuits]

    # 2. Execute
    job = estimator.run(pubs)
    result = job.result()

    # 3. Extract and scale EVs
    if is_blueprint:
        evs = result[0].data.evs
    else:
        evs = np.array([res.data.evs for res in result])

    num_terms = len(observable.paulis)
    if len(observable.paulis) > 1:
        evs = evs / num_terms

    # 4. Generate predictions based on loss_type
    if loss_type == "mse":
        predictions = np.where(evs >= 0.0, 0, 1)
    elif loss_type == "cross_entropy":
        probs = np.clip((1.0 - evs) / 2.0, 1e-15, 1 - 1e-15)
        predictions = np.where(probs >= 0.5, 1, 0)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return predictions, evs
