import argparse
import datetime
import json
import os
import time
import uuid
import warnings

import numpy as np
import yaml
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from scipy.optimize import Bounds
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src import factories
from src.data_loader import load_data, scale_data
from src.engine import predict_vqc, train_vqc

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    start_time = time.time()

    # 1. Parse the arguments
    parser = argparse.ArgumentParser(description="Execute a single VQC simulation")
    parser.add_argument(
        "--fmap", type=str, required=True, help="Specify the feature map name"
    )
    parser.add_argument(
        "--ansatz", type=str, required=True, help="Specify the ansatz name"
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Specify the random seed"
    )
    parser.add_argument(
        "--fmap_reps",
        type=int,
        default=1,
        help="Specify the number of repetitions for the feature map",
    )
    parser.add_argument(
        "--ansatz_reps",
        type=int,
        default=1,
        help="Specify the number of repetitions for the ansatz",
    )
    parser.add_argument(
        "--fmap_entanglement",
        type=str,
        default="linear",
        help="Specify the feature map entanglement topology",
    )
    parser.add_argument(
        "--ansatz_entanglement",
        type=str,
        default="linear",
        help="Specify the feature map entanglement topology",
    )
    args = parser.parse_args()
    np.random.seed(args.seed)

    # 2. Load configuration
    with open("configs/vqc_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(
        f">>> Start task: FMAP={args.fmap} | ANSATZ={args.ansatz} | SEED={args.seed} | FMAP_REPS={args.fmap_reps} | ANSATZ_REPS={args.ansatz_reps} | FMAP_ENTANGLEMENT={args.fmap_entanglement} | ANSATZ_ENTANGLEMENT={args.ansatz_entanglement} | OBSERVABLE={config['observable']}"
    )

    # 3. Get backend and pass manager
    backend_type = config.get("backend", {}).get("type", "LOCAL").upper()
    backend = factories.get_backend(config)
    print(f">>> Using backend: {backend.name}")

    print(">>> Generating pass manager with optimization level 1...")
    pm = generate_preset_pass_manager(
        optimization_level=1, backend=backend, seed_transpiler=args.seed
    )
    assert pm is not None, "Pass manager generation failed!"

    # 4. Prepare train/test subsets and weights for the classes based on the config
    # Split the dataset with a stratfied approach and create weights list
    X, Y = load_data(config["dataset_path"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.25, random_state=args.seed, stratify=Y
    )
    X_train, X_test = scale_data(
        X_train=X_train, X_test=X_test, feature_range=tuple(config["feature_range"])
    )

    weights_dict = config[
        "weights"
    ]  # Dictionary of weights for the classes in the dataset
    weights = np.array([weights_dict[str(int(label))] for label in y_train])

    # 5. Build circuits and transpile
    factory = factories.CircuitFactory(
        num_features=X_train.shape[1],
        fmap_name=args.fmap,
        fmap_reps=args.fmap_reps,
        fmap_entanglement=args.fmap_entanglement,
        ansatz_name=args.ansatz,
        ansatz_reps=args.ansatz_reps,
        ansatz_entanglement=args.ansatz_entanglement,
    )

    circuits_train, is_blueprint = factory.build(
        data=X_train if args.fmap == "Amplitude" else None
    )

    print(">>> Transpiling circuits...")
    circuits_train = pm.run(circuits_train)

    # 6. Prepare the rest of the necessary components:
    # bounds, observable and estimator

    bound_range = tuple(
        config["theta_range"]
    )  # Range for the parameters (theta) of the VQC
    bounds = Bounds(bound_range[0], bound_range[1])

    observable = factories.get_observable(
        observable_name=config["observable"], num_qubits=factory.num_qubits
    )

    estimator = factories.get_estimator(backend=backend, config=config, seed=args.seed)

    # Initialize parameters randomly within the specified theta range
    initial_params = np.random.uniform(
        config["theta_range"][0], config["theta_range"][1], size=factory.num_weights
    )
    # 7. Apply layout if pm is not None
    # It ensures that the observable is correctly mapped to the qubits after transpilation

    layout_circ = circuits_train if is_blueprint else circuits_train[0]
    if layout_circ.layout is not None:
        observable = observable.apply_layout(
            layout_circ.layout, num_qubits=layout_circ.num_qubits
        )

    # 8. Start VQC training
    result_params, loss_history, accuracy_history = train_vqc(
        circuits=circuits_train,
        is_blueprint=is_blueprint,
        X_train=X_train,
        y_train=y_train,
        observable=observable,
        num_weights=factory.num_weights,
        initial_params=initial_params,
        config=config,
        sample_weights=weights,
        estimator=estimator,
        bounds=bounds,
    )

    # 9. Prepare test circuits
    if is_blueprint:
        circuits_test = circuits_train
    else:
        print(">>> Generating specific test circuits (bound data mode)...")
        circuits_test, _ = factory.build(
            data=X_test if args.fmap == "Amplitude" else None
        )
        circuits_test = pm.run(circuits_test)

    # 10. Evaluate the model on both train and test sets

    print(">>> Evaluating model on Train and Test sets...")
    pos_label = 1

    # Train subset predictions
    train_preds, train_evs = predict_vqc(
        circuits_train,
        is_blueprint,
        X_train,
        result_params,
        observable,
        factory.num_weights,
        config,
        estimator,
    )
    train_acc = accuracy_score(y_train, train_preds)

    # Test subset predictions
    test_preds, test_expectations = predict_vqc(
        circuits_test,
        is_blueprint,
        X_test,
        result_params,
        observable,
        factory.num_weights,
        config,
        estimator,
    )
    test_acc = accuracy_score(y_test, test_preds)

    # 11. Calculate additional metrics: F1, Recall, Precision, Specificity, ROC and AUC
    f1 = f1_score(y_test, test_preds, pos_label=pos_label, zero_division=0)
    recall = recall_score(y_test, test_preds, pos_label=pos_label, zero_division=0)
    precision = precision_score(
        y_test, test_preds, pos_label=pos_label, zero_division=0
    )

    # Confusion matrix and specificity
    cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # ROC CURVE
    scores_for_roc = test_expectations
    fpr, tpr, thresholds = roc_curve(y_test, scores_for_roc, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    end_time = time.time()
    elapsed_time_seconds = end_time - start_time

    print(
        f">>> Training & Evaluation completed - time: {elapsed_time_seconds:.2f} s | "
        f"Test Acc: {test_acc * 100:.2f}% | Train Acc: {train_acc * 100:.2f}% | F1: {f1:.3f} | Recall: {recall:.3f} | Spec: {specificity:.3f} | Prec: {precision:.3f} | AUC: {roc_auc:.3f}"
    )

    # 12. Save the results to a JSON file
    entry = {
        # CONFIG
        "CONFIG": {
            "dataset_path": config["dataset_path"],
            "backend": {"type": backend_type, "name": backend.name},
            "optimizer": {
                "name": config["optimizer"]["name"],
                "num_epochs": config["optimizer"]["num_epochs"],
                "tol": config["optimizer"]["tol"],
                "rho_beg": config["optimizer"]["rho_beg"],
            },
            "theta_range": config["theta_range"],
            "observable": config["observable"],
            "estimator_shots": config["estimator_shots"],
            "loss_type": config["loss_type"],
            "weights": config["weights"],
        },
        # ARCHITECTURE
        "ARCHITECTURE": {
            "seed": args.seed,
            "fmap": args.fmap,
            "fmap_reps": args.fmap_reps,
            "fmap_entanglement": args.fmap_entanglement,
            "ansatz": args.ansatz,
            "ansatz_reps": args.ansatz_reps,
            "ansatz_entanglement": args.ansatz_entanglement,
            "num_qubits": factory.num_qubits,
            "num_params": factory.num_weights,
            "num_features": factory.num_features,
        },
        # RESULTS
        "RESULTS": {
            "execution_time_seconds": elapsed_time_seconds,
            "training_history": {"loss": loss_history, "accuracy": accuracy_history},
            "final_train_accuracy": train_acc,
            "final_test_accuracy": test_acc,
            "test_f1": f1,
            "recall": recall,
            "precision": precision,
            "specificity": specificity,
            "confusion_matrix": cm.tolist(),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": roc_auc,
            "initial_params": initial_params.tolist(),
            "best_params": result_params.tolist(),
        },
    }

    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M")
    run_id = uuid.uuid4().hex[:6]

    fmap_short = args.fmap[:8]
    ansatz_short = args.ansatz[:8]
    fmap_ent_short = args.fmap_entanglement[:4]
    ansatz_ent_short = args.ansatz_entanglement[:4]
    filename = f"{timestamp}_seed{args.seed}_{fmap_short}_ent_{fmap_ent_short}_{ansatz_short}_ent_{ansatz_ent_short}_{run_id}.json"

    dataset_name = os.path.basename(config["dataset_path"]).split(".")[0]
    backend_name = backend.name

    save_dir = os.path.join("results", backend_name, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    full_path = os.path.join(save_dir, filename)

    with open(full_path, "w") as f:
        json.dump(entry, f, indent=4)

    print(f">>> End of task. Saved to: {full_path}")
