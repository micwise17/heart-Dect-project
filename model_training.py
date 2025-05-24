# model_training.py

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd

# Import functions from previous modules
from data_preprocessing import load_data, preprocess_data
from feature_selection import get_mi_scores

def tune_knn_hyperparameters(X_train, y_train, k_range=range(3, 21, 2), n_splits=10, random_state=42):
    """Tunes the K hyperparameter for KNN using GridSearchCV.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        k_range (range, optional): Range of K values to test. Defaults to odd numbers 1-19.
        n_splits (int, optional): Number of folds for cross-validation. Defaults to 10.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: (best_k, best_knn_estimator) found by GridSearchCV.
    """
    print(f"Starting KNN hyperparameter tuning (K) using {n_splits}-fold CV...")
    print(f"Testing K values in: {list(k_range)}")

    # Define the parameter grid for K
    param_grid = {"n_neighbors": k_range}

    # Initialize KNN classifier (using Euclidean distance as per report)
    knn = KNeighborsClassifier(metric="euclidean")

    # Setup Stratified K-Fold cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Setup GridSearchCV (optimize for F1-score as per report)
    # n_jobs=-1 uses all available CPU cores
    grid_search = GridSearchCV(knn, param_grid, cv=cv, scoring="f1", n_jobs=-1)

    # Fit GridSearchCV to the training data
    grid_search.fit(X_train, y_train)

    # Get the best K value and the best estimator
    best_k = grid_search.best_params_["n_neighbors"]
    best_knn_estimator = grid_search.best_estimator_

    print(f"Hyperparameter tuning complete. Best K found: {best_k}")
    print(f"Best F1 score during CV: {grid_search.best_score_:.4f}")

    return best_k, best_knn_estimator

# Example usage (can be run as a script)
if __name__ == "__main__":
    print("Running Model Training Module...")

    # 1. Load and preprocess data
    raw_data = load_data() # Assumes "heart.csv" is present
    if raw_data is not None:
        processed_data = preprocess_data(raw_data)
        if processed_data:
            X_train, X_test, y_train, y_test, feature_names = processed_data

            # --- Scenario 1: Train KNN with Full Features ---
            print("\n--- Training KNN with Full Features ---")
            best_k_full, knn_model_full = tune_knn_hyperparameters(X_train, y_train)
            print(f"Tuned KNN model (Full Features): {knn_model_full}")

            # --- Scenario 2: Train KNN with MI-Selected Features ---
            print("\n--- Training KNN with MI-Selected Features ---")
            # 2a. Get MI scores and select features
            ranked_features = get_mi_scores(X_train, y_train, feature_names)
            top_k_features = 8 # As determined in the report methodology/results
            selected_features_mi = ranked_features.head(top_k_features).index.tolist()
            print(f"Using top {top_k_features} features selected by MI: {selected_features_mi}")

            # 2b. Filter training data to selected features
            X_train_mi = X_train[selected_features_mi]

            # 2c. Tune and train KNN on MI-selected features
            best_k_mi, knn_model_mi = tune_knn_hyperparameters(X_train_mi, y_train)
            print(f"Tuned KNN model (MI Features): {knn_model_mi}")

            # The trained models (knn_model_full, knn_model_mi) and the corresponding
            # test sets (X_test, X_test[selected_features_mi]) can now be passed
            # to the evaluation module.

        else:
            print("Data preprocessing failed. Cannot perform model training.")
    else:
        print("Data loading failed. Cannot perform model training.")

