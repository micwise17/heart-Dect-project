# feature_selection.py

import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# Import functions from the data preprocessing module
# This assumes data_preprocessing.py is in the same directory
from data_preprocessing import load_data, preprocess_data

def get_mi_scores(X_train, y_train, feature_names, discrete_features="auto", random_state=42):
    """Calculates and ranks features based on Mutual Information scores.

    Args:
        X_train (pd.DataFrame): Preprocessed training features.
        y_train (pd.Series): Training target variable.
        feature_names (list): List of feature names corresponding to X_train columns.
        discrete_features (str or array-like, optional): Specifies which features are discrete.
            Defaults to "auto". See scikit-learn documentation for details.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        pd.Series: Features ranked by their MI scores (descending order).
    """
    print("Calculating Mutual Information scores...")
    # Ensure correct data types for MI calculation
    # MI works best with discrete features or binned continuous ones.
    # 'auto' tries to handle mixed types, but results can vary.
    # Consider explicitly marking categorical features if 'auto' is insufficient.
    # For this dataset, 'auto' often works reasonably well.
    mi_scores = mutual_info_classif(
        X_train, y_train,
        discrete_features=discrete_features,
        random_state=random_state
    )

    # Create a pandas Series for better visualization and ranking
    mi_series = pd.Series(mi_scores, index=feature_names)

    # Sort features by MI score in descending order
    mi_series_sorted = mi_series.sort_values(ascending=False)

    print("Mutual Information scores calculated and ranked.")
    return mi_series_sorted

# Example usage (can be run as a script)
if __name__ == "__main__":
    print("Running Feature Selection Module...")
    # 1. Load and preprocess data using the other module
    raw_data = load_data() # Assumes 'heart.csv' is present
    if raw_data is not None:
        processed_data = preprocess_data(raw_data)
        if processed_data:
            X_train, X_test, y_train, y_test, feature_names = processed_data

            # 2. Calculate MI scores
            ranked_features = get_mi_scores(X_train, y_train, feature_names)

            print("\nMutual Information Scores (Ranked):")
            print(ranked_features)

            # Example: Select top 8 features as suggested in the report
            top_k = 8
            selected_features_mi = ranked_features.head(top_k).index.tolist()
            print(f"\nExample: Top {top_k} features based on MI:")
            print(selected_features_mi)

            # These selected_features_mi can be used to filter X_train and X_test
            # X_train_mi = X_train[selected_features_mi]
            # X_test_mi = X_test[selected_features_mi]
            # print("\nShape of X_train after MI selection:", X_train_mi.shape)

        else:
            print("Data preprocessing failed. Cannot perform feature selection.")
    else:
        print("Data loading failed. Cannot perform feature selection.")

