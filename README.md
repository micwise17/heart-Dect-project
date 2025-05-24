# Heart Disease Diagnosis MI-KNN Code

This package contains Python scripts to reproduce the analysis described in the undergraduate project report "Development of Heart Disease Diagnostic System Using Mutual Information Feature Selection and K-Nearest Neighbour Algorithms".

## Files Included

- `data_preprocessing.py`: Loads the dataset, handles missing values, encodes the target, splits data, and scales features.
- `feature_selection.py`: Calculates Mutual Information scores and ranks features.
- `model_training.py`: Tunes the K hyperparameter for KNN using GridSearchCV and trains the final models (one with full features, one with MI-selected features).
- `evaluation.py`: Evaluates the trained models on the test set, generates performance metrics, and creates plots (MI scores, confusion matrices, ROC curves) saved in the `appendix_outputs` directory.
- `README.md`: This file.

## Requirements

- Python 3.9 or higher.
- The following Python libraries:
    - pandas
    - numpy
    - scikit-learn
    - matplotlib
    - seaborn

You can install these using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Dataset

- You need to obtain the Heart Disease dataset, commonly available from the UCI Machine Learning Repository or platforms like Kaggle.
- The expected format is a CSV file (e.g., `heart.csv`) containing the 14 standard attributes used in most research (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target).
- **Crucially, place the `heart.csv` file (or rename your dataset file to `heart.csv`) in the SAME directory where you extract these Python scripts.** The `data_preprocessing.py` script looks for `heart.csv` by default.

## How to Run

1.  Ensure you have Python and the required libraries installed.
2.  Download and place the `heart.csv` dataset file in the same directory as the scripts.
3.  Open a terminal or command prompt in that directory.
4.  Run the main evaluation script:
    ```bash
    python evaluation.py
    ```

This single command will execute the entire pipeline:
- Load and preprocess the data (`data_preprocessing.py` functions are called).
- Calculate MI scores and plot them (`feature_selection.py` functions are called).
- Train and tune both KNN models (full features and MI-selected features) (`model_training.py` functions are called).
- Evaluate both models on the test set, print metrics to the console, and save the performance summary table (`table_c1_performance_metrics.csv`) and plots (`figure_c*.png`) into a new sub-directory named `appendix_outputs`.

## Notes

- The scripts assume the standard 14 features and target encoding (0=absence, 1-4=presence, converted to 0/1) as described in the report.
- The number of MI features selected (`top_k = 8`) is hardcoded in `model_training.py` and `evaluation.py` based on the report's findings. You can modify this value if needed.
- The output plots and table correspond to the placeholders mentioned in Appendix C of the project report.

