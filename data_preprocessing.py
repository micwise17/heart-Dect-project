# data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Define expected column names based on standard UCI Heart Disease dataset (14 features)
EXPECTED_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Define numerical and categorical features (based on common usage)
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COLUMN = "target"

def load_data(filepath="heart.csv"):
    """Loads the heart disease dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data or None if file not found/invalid.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from {filepath}")
        # Basic validation: Check if expected columns are present
        if not all(col in df.columns for col in EXPECTED_COLUMNS):
            print(f"Error: Dataset at {filepath} is missing expected columns.")
            print(f"Expected columns: {EXPECTED_COLUMNS}")
            print(f"Found columns: {df.columns.tolist()}")
            return None
        return df
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {filepath}")
        print("Please ensure the heart disease dataset (e.g., 'heart.csv') is in the same directory as the scripts.")
        return None
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None

def preprocess_data(df):
    """Preprocesses the heart disease data.

    Handles missing values, encodes target, splits data, and scales features.

    Args:
        df (pd.DataFrame): The raw dataframe.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names) if successful, else None.
    """
    print("Starting data preprocessing...")

    # 1. Handle potential missing value markers (like "?")
    # Replace "?" with NaN if present, and convert relevant columns to numeric
    df = df.replace("?", np.nan)
    # Ensure columns that should be numeric are treated as such, coercing errors
    # Specifically 'ca' and 'thal' are often problematic in raw UCI data
    for col in ["ca", "thal"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. Impute missing values (using median for numerical as per report)
    # Note: The standard 14-feature processed Cleveland dataset often has few/no NaNs,
    # but imputation is included for robustness based on report methodology.
    missing_cols = df.isnull().sum()
    if missing_cols.sum() > 0:
        print("Missing values found. Imputing...")
        print(missing_cols[missing_cols > 0])
        for col in NUMERICAL_FEATURES:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Imputed missing values in '{col}' with median: {median_val}")
        # Impute categorical with mode (less common in the 14-feature set)
        for col in CATEGORICAL_FEATURES:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                print(f"Imputed missing values in '{col}' with mode: {mode_val}")
        # Check if any NaNs remain after imputation
        if df.isnull().sum().sum() > 0:
            print("Warning: Missing values still remain after imputation!")
            print(df.isnull().sum()[df.isnull().sum() > 0])
            # Consider dropping rows/cols or using more advanced imputation if critical
            # For now, we proceed, but this might cause issues later.
    else:
        print("No missing values found.")

    # 3. Encode target variable (0 = no disease, 1 = presence of disease)
    # UCI standard: 0 = absence, 1, 2, 3, 4 = presence
    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(lambda x: 1 if x > 0 else 0)
    print(f"Target variable '{TARGET_COLUMN}' encoded to binary (0/1). Value counts:")
    print(df[TARGET_COLUMN].value_counts())

    # 4. Separate features (X) and target (y)
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    feature_names = X.columns.tolist()
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")

    # Ensure all feature columns are numeric before scaling/splitting
    # This might involve one-hot encoding categoricals if needed by the model,
    # but KNN with Euclidean distance primarily relies on numerical inputs.
    # For simplicity here, we assume categorical features are numerically encoded
    # (as they are in the standard UCI dataset: sex 0/1, cp 1-4, etc.)
    # and only scale the explicitly numerical ones.
    try:
        X = X.astype(float)
    except ValueError as e:
        print(f"Error converting features to float: {e}")
        print("Ensure all feature columns are numeric or appropriately encoded.")
        return None

    # 5. Split data into training and testing sets (75% train, 25% test, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Data split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples).")

    # 6. Scale numerical features using StandardScaler
    scaler = StandardScaler()
    # Fit scaler ONLY on training data numerical features
    X_train_scaled_num = scaler.fit_transform(X_train[NUMERICAL_FEATURES])
    # Transform test data numerical features using the SAME fitted scaler
    X_test_scaled_num = scaler.transform(X_test[NUMERICAL_FEATURES])

    # Create DataFrames for scaled numerical features
    X_train_scaled_num_df = pd.DataFrame(X_train_scaled_num, columns=NUMERICAL_FEATURES, index=X_train.index)
    X_test_scaled_num_df = pd.DataFrame(X_test_scaled_num, columns=NUMERICAL_FEATURES, index=X_test.index)

    # Combine scaled numerical features with original categorical features
    X_train_processed = pd.concat([X_train_scaled_num_df, X_train[CATEGORICAL_FEATURES]], axis=1)
    X_test_processed = pd.concat([X_test_scaled_num_df, X_test[CATEGORICAL_FEATURES]], axis=1)

    # Reorder columns to original order for consistency
    X_train_processed = X_train_processed[feature_names]
    X_test_processed = X_test_processed[feature_names]

    print("Numerical features scaled using StandardScaler.")
    print("Preprocessing complete.")

    return X_train_processed, X_test_processed, y_train, y_test, feature_names

# Example usage (can be run as a script)
if __name__ == "__main__":
    print("Running Data Loading and Preprocessing Module...")
    raw_data = load_data() # Assumes 'heart.csv' in the same directory
    if raw_data is not None:
        processed_data = preprocess_data(raw_data)
        if processed_data:
            X_train, X_test, y_train, y_test, features = processed_data
            print("\nPreprocessing finished successfully.")
            print("X_train shape:", X_train.shape)
            print("X_test shape:", X_test.shape)
            print("y_train distribution:\n", y_train.value_counts(normalize=True))
            print("y_test distribution:\n", y_test.value_counts(normalize=True))
            # Display first 5 rows of processed training data
            print("\nFirst 5 rows of processed X_train:")
            print(X_train.head())
        else:
            print("Data preprocessing failed.")
    else:
        print("Data loading failed.")

