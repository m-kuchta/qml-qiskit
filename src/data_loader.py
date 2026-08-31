import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads a dataset from a CSV file and separates features from the target variable.

    Args:
        filepath (str): The path to the CSV dataset file.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the feature matrix (X)
        and the target vector (Y).
    """
    dataset = pd.read_csv(filepath)

    # Separate features and target label (assuming "Outcome" is the target column)
    X = dataset.drop(columns=["Outcome"])
    Y = dataset["Outcome"]

    X = np.array(X)
    Y = np.array(Y)

    return (X, Y)


def scale_data(
    X_train: np.ndarray, X_test: np.ndarray, feature_range: tuple
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scales features to a specified range using MinMaxScaler.

    Args:
        X_train (np.ndarray): Training feature matrix.
        X_test (np.ndarray): Testing feature matrix.
        feature_range (tuple): Desired range of transformed data (e.g., (-1, 1)).

    Returns:
        tuple: A tuple containing the scaled training and testing features
        (X_train_scaled, X_test_scaled).
    """

    scaler = MinMaxScaler(feature_range=feature_range)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled
