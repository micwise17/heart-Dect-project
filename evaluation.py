# evaluation.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_curve
)
from sklearn.neighbors import KNeighborsClassifier # Import KNeighborsClassifier
import os
import time

# Import functions/logic from previous modules
from data_preprocessing import load_data, preprocess_data
from feature_selection import get_mi_scores
# from model_training import tune_knn_hyperparameters # No longer tuning K

# Define output directory for plots and tables
OUTPUT_DIR = "appendix_outputs"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluates a trained model on the test set and returns metrics.

    Args:
        model: The trained scikit-learn classifier.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True test labels.
        model_name (str): Name of the model for printing.

    Returns:
        dict: Dictionary containing performance metrics.
        tuple: (y_pred, y_proba) predictions and probabilities.
        float: Prediction time in seconds.
    """
    print(f"\n--- Evaluating {model_name} on Test Set ---")
    # Measure prediction time
    start_time = time.time()
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        print("Warning: Model does not support predict_proba. AUC cannot be calculated.")
        y_proba = None
    end_time = time.time()
    prediction_time = end_time - start_time
    print(f"Prediction Time: {prediction_time:.4f} seconds")

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred) # Sensitivity
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if auc is not None:
        print(f"AUC: {auc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1-Score": f1,
        "AUC": auc
    }
    return metrics, (y_pred, y_proba), prediction_time

def plot_mi_scores(mi_scores_ranked, filename="figure_4.1_mi_scores.png"):
    """Plots and saves the ranked Mutual Information scores.

    Args:
        mi_scores_ranked (pd.Series): Ranked MI scores.
        filename (str): Output filename for the plot.
    """
    print(f"Plotting MI scores to {filename}...")
    plt.figure(figsize=(10, 6))
    mi_scores_ranked.plot(kind="bar")
    plt.title("Figure 4.1: Mutual Information Scores for Features")
    plt.ylabel("MI Score")
    plt.xlabel("Features")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print("MI scores plot saved.")

def save_mi_scores_table(mi_scores_ranked, feature_descriptions, filename="table_4.1_mi_scores.csv"):
    """Saves the ranked MI scores and descriptions to a CSV file.

    Args:
        mi_scores_ranked (pd.Series): Ranked MI scores.
        feature_descriptions (dict): Dictionary mapping feature names to descriptions.
        filename (str): Output filename for the table.
    """
    print(f"Saving MI scores table to {filename}...")
    mi_df = pd.DataFrame({
        "Feature": mi_scores_ranked.index,
        "MI Score": mi_scores_ranked.values
    })
    mi_df["Rank"] = range(1, len(mi_df) + 1)
    mi_df["Description"] = mi_df["Feature"].map(feature_descriptions)
    mi_df = mi_df[["Rank", "Feature", "MI Score", "Description"]]
    # Format MI Score to 4 decimal places for consistency in the table
    mi_df["MI Score"] = mi_df["MI Score"].map("{:.4f}".format)
    mi_df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
    print("MI scores table saved.")

def plot_confusion_matrix(y_true, y_pred, model_name="Model", filename="confusion_matrix.png"):
    """Plots and saves the confusion matrix.

    Args:
        y_true (pd.Series): True labels.
        y_pred (pd.Series): Predicted labels.
        model_name (str): Name of the model for the title.
        filename (str): Output filename for the plot.
    """
    print(f"Plotting confusion matrix for {model_name} to {filename}...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print("Confusion matrix plot saved.")

def plot_roc_curves(results, filename="figure_4.4_roc_curves.png"):
    """Plots ROC curves for multiple models.

    Args:
        results (dict): Dictionary where keys are model names and values are
                        tuples (y_test, y_proba).
        filename (str): Output filename for the plot.
    """
    print(f"Plotting ROC curves to {filename}...")
    plt.figure(figsize=(8, 6))
    for name, (y_test, y_proba) in results.items():
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = roc_auc_score(y_test, y_proba)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")
        else:
            print(f"Cannot plot ROC for {name} as probabilities are unavailable.")

    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)")  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Figure 4.4: Receiver Operating Characteristic (ROC) Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print("ROC curves plot saved.")

# Main execution block
if __name__ == "__main__":
    print("Running Evaluation and Visualization Module...")

    # Feature descriptions (as per UCI documentation)
    feature_descriptions = {
        "age": "Age in years",
        "sex": "Sex (1=male; 0=female)",
        "cp": "Chest pain type",
        "trestbps": "Resting blood pressure in mm Hg",
        "chol": "Serum cholesterol in mg/dl",
        "fbs": "Fasting blood sugar > 120 mg/dl",
        "restecg": "Resting electrocardiographic results",
        "thalach": "Maximum heart rate achieved",
        "exang": "Exercise induced angina",
        "oldpeak": "ST depression induced by exercise relative to rest",
        "slope": "Slope of the peak exercise ST segment",
        "ca": "Number of major vessels colored by fluoroscopy",
        "thal": "Thallium stress test result"
    }

    # 1. Load and preprocess data (using random_state=42 from data_preprocessing.py)
    raw_data = load_data() # Assumes "heart.csv" is present
    if raw_data is None:
        exit("Data loading failed. Exiting.")

    processed_data = preprocess_data(raw_data)
    if processed_data is None:
        exit("Data preprocessing failed. Exiting.")

    X_train, X_test, y_train, y_test, feature_names = processed_data

    # 2. Get MI Scores (for plotting and table)
    ranked_features = get_mi_scores(X_train, y_train, feature_names)
    plot_mi_scores(ranked_features, filename="figure_4.1_mi_scores.png")
    save_mi_scores_table(ranked_features, feature_descriptions, filename="table_4.1_mi_scores.csv")

    # 3. Train Models with FIXED K values (K=7 for full, K=9 for MI)
    # KNN with Full Features (K=7)
    print("\nTraining KNN (Full Features) with K=7...")
    knn_model_full = KNeighborsClassifier(n_neighbors=7)
    knn_model_full.fit(X_train, y_train)
    print("KNN (Full Features, K=7) trained.")

    # KNN with MI-Selected Features (K=9)
    print("\nTraining KNN (MI Features) with K=9...")
    top_k_features = 8
    selected_features_mi = ranked_features.head(top_k_features).index.tolist()
    print(f"Selected Features (Top {top_k_features}): {selected_features_mi}")
    X_train_mi = X_train[selected_features_mi]
    X_test_mi = X_test[selected_features_mi]
    knn_model_mi = KNeighborsClassifier(n_neighbors=9)
    knn_model_mi.fit(X_train_mi, y_train)
    print("KNN (MI Features, K=9) trained.")

    # 4. Evaluate Models
    metrics_full, (y_pred_full, y_proba_full), time_full = evaluate_model(
        knn_model_full, X_test, y_test, model_name="KNN (Full Features, K=7)"
    )
    metrics_mi, (y_pred_mi, y_proba_mi), time_mi = evaluate_model(
        knn_model_mi, X_test_mi, y_test, model_name="KNN (MI Features, K=9)"
    )

    # 5. Create Performance Summary Table (Table 4.2)
    metrics_df = pd.DataFrame({
        "KNN (Full Features, K=7)": metrics_full,
        "KNN (MI Features, K=9)": metrics_mi
    })
    # Add prediction time
    metrics_df.loc["Prediction Time (s)"] = [time_full, time_mi]
    # Calculate difference
    metrics_df["Difference"] = metrics_df["KNN (MI Features, K=9)"] - metrics_df["KNN (Full Features, K=7)"]

    print("\n--- Performance Summary Table (Table 4.2) ---")
    # Create a copy for formatted output
    metrics_df_formatted = metrics_df.copy()

    # Format metrics as percentages, except for Prediction Time
    metrics_to_format = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "AUC"]
    for metric in metrics_to_format:
        if metric in metrics_df_formatted.index:
            metrics_df_formatted.loc[metric] = metrics_df_formatted.loc[metric].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")

    # Format Prediction Time to 4 decimal places
    if "Prediction Time (s)" in metrics_df_formatted.index:
        metrics_df_formatted.loc["Prediction Time (s)"] = metrics_df_formatted.loc["Prediction Time (s)"].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")

    # Format Difference column (percentage for metrics, decimal for time)
    for metric in metrics_to_format:
         if metric in metrics_df_formatted.index:
             # Use original numeric difference for calculation
             diff_val = metrics_df.loc[metric, "Difference"]
             metrics_df_formatted.loc[metric, "Difference"] = f"{diff_val*100:+.2f}%" if pd.notnull(diff_val) else "N/A"
    if "Prediction Time (s)" in metrics_df_formatted.index:
        diff_val = metrics_df.loc["Prediction Time (s)", "Difference"]
        metrics_df_formatted.loc["Prediction Time (s)", "Difference"] = f"{diff_val:+.4f}" if pd.notnull(diff_val) else "N/A"

    print(metrics_df_formatted)

    # Save formatted table to CSV
    table_filename = os.path.join(OUTPUT_DIR, "table_4.2_performance_metrics.csv")
    metrics_df_formatted.to_csv(table_filename)
    print(f"Formatted performance table saved to {table_filename}")

    # 6. Generate Plots (Figures 4.2, 4.3, 4.4)
    # Confusion Matrices
    plot_confusion_matrix(y_test, y_pred_full, model_name="KNN (Full Features, K=7)",
                          filename="figure_4.2_cm_full_k7.png") # Updated filename
    plot_confusion_matrix(y_test, y_pred_mi, model_name="KNN (MI Features, K=9)",
                          filename="figure_4.3_cm_mi_k9.png") # Updated filename

    # ROC Curves
    roc_results = {
        "KNN (Full Features, K=7)": (y_test, y_proba_full),
        "KNN (MI Features, K=9)": (y_test, y_proba_mi)
    }
    plot_roc_curves(roc_results, filename="figure_4.4_roc_curves_k7k9.png") # Updated filename

    print("\nEvaluation and visualization complete.")
    print(f"Outputs saved in directory: {OUTPUT_DIR}")

