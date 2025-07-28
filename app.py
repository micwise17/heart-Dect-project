import os
import warnings
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from data_preprocessing import load_data, preprocess_data
from feature_selection import get_mi_scores
from evaluation import evaluate_model

 # Define output directory
OUTPUT_DIR = "appendix_outputs"

st.title("Heart Disease KNN MI Dashboard")

# 1. Load and preprocess data
st.sidebar.header("Data Preprocessing")
raw_data = load_data()
if raw_data is None:
    st.error("Failed to load data. Ensure heart.csv is present.")
    st.stop()
processed = preprocess_data(raw_data)
if processed is None:
    st.error("Data preprocessing failed.")
    st.stop()
X_train, X_test, y_train, y_test, feature_names = processed

st.sidebar.success("Data loaded and preprocessed.")

#  # Show dataset overview
st.subheader("Dataset Overview")
st.write("### First 5 rows:")
st.dataframe(raw_data.head())
st.write(f"Data shape: {raw_data.shape}")

# 2. Mutual Information Scores
st.sidebar.header("Feature Selection")
mi_scores = get_mi_scores(X_train, y_train, feature_names)
top_k = st.sidebar.slider("Select top K features (MI)", min_value=1, max_value=len(feature_names), value=8)
selected_features = mi_scores.head(top_k).index.tolist()

st.subheader("Mutual Information Scores")
st.write(mi_scores.to_frame(name="MI Score"))

# Plot MI scores
if st.sidebar.checkbox("Show MI Scores Plot"):
    fig, ax = st.pyplot()
    mi_scores.plot(kind="bar", figsize=(10,4), ax=ax)
    ax.set_title("Mutual Information Scores")
    ax.set_ylabel("MI Score")
    ax.set_xlabel("Features")

# 3. Model Training
st.sidebar.header("Model Training")

# Fixed K values
k_full = st.sidebar.number_input("K for Full Features", min_value=1, max_value=100, value=7)
k_mi = st.sidebar.number_input("K for MI Features", min_value=1, max_value=100, value=9)

if st.sidebar.button("Run Evaluation"):
    # Train full model
    knn_full = KNeighborsClassifier(n_neighbors=k_full)
    knn_full.fit(X_train, y_train)
    metrics_full, (y_pred_full, y_proba_full), _ = evaluate_model(knn_full, X_test, y_test, model_name=f"KNN Full (K={k_full})")

    # Train MI model
    knn_mi = KNeighborsClassifier(n_neighbors=k_mi)
    knn_mi.fit(X_train[selected_features], y_train)
    metrics_mi, (y_pred_mi, y_proba_mi), _ = evaluate_model(knn_mi, X_test[selected_features], y_test, model_name=f"KNN MI (K={k_mi})")

    # Display metrics
    st.subheader("Performance Metrics")
    df_metrics = pd.DataFrame({f"Full K={k_full}": metrics_full, f"MI K={k_mi}": metrics_mi})
    st.dataframe(df_metrics)

    # Confusion Matrices
    st.subheader("Confusion Matrices")
    cm1 = confusion_matrix(y_test, y_pred_full)
    cm2 = confusion_matrix(y_test, y_pred_mi)
    st.write(f"KNN Full (K={k_full})")
    st.write(cm1)
    st.write(f"KNN MI (K={k_mi})")
    st.write(cm2)

    # ROC Curves
    st.subheader("ROC Curves")
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib.pyplot as plt
    fig2, ax2 = plt.subplots()
    for name, (y_true, y_prob) in [(f"Full K={k_full}", (y_test, y_proba_full)), (f"MI K={k_mi}", (y_test, y_proba_mi))]:
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            ax2.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, y_prob):.2f})")
    ax2.plot([0,1],[0,1], 'k--')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend()
    st.pyplot(fig2)

# 4. Reports & Outputs
st.sidebar.header("Reports & Outputs")
if st.sidebar.checkbox("Show Saved Reports and Images"):
    files = sorted(os.listdir(OUTPUT_DIR))
    for file in files:
        file_path = os.path.join(OUTPUT_DIR, file)
        if file.lower().endswith(".png"):
            st.subheader(file)
            st.image(file_path, use_container_width=True)
        elif file.lower().endswith(".csv"):
            st.subheader(file)
            try:
                df_file = pd.read_csv(file_path)
                st.dataframe(df_file)
            except Exception as e:
                st.write(f"Failed to load {file}: {e}")
