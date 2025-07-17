"""
This script runs a cross-validation experiment on a collection of datasets in parallel.

For each dataset, it performs the following steps for each CV fold:
1. Splits the data into training and testing sets.
2. Creates a preprocessing pipeline (StandardScaler, VarianceThreshold, PCA).
3. Fits the pipeline on the training data ONLY.
4. Transforms both training and testing data using the fitted pipeline.
5. Fits a custom classifier and measures performance metrics.

Metrics (accuracy, time, memory) are averaged across all folds and saved to a CSV file.

Usage:
    python3 experiment.py /path/to/datasets_folder /path/to/output.csv
"""

import argparse
import time
import tracemalloc
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from classifier.knn import BaselineKNNClassifier


def process_dataset(file_path: Path) -> dict | None:
    """
    Processes a single dataset file. It runs cross-validation and collects metrics.

    Parameters
    ----------
        file_path (Path): The path to the input CSV dataset.

    Returns
    -------
        A dictionary containing the aggregated results for the dataset, or None on error.
    """
    dataset_name = file_path.stem
    print(f"🚀 Starting processing for dataset: {dataset_name}")

    try:
        # 1. Load and prepare data
        df = pd.read_csv(file_path)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # 2. Define the cross-validation strategy
        # Using StratifiedKFold is good practice for classification tasks
        n_splits = 10
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

        # 3. Initialize lists to store metrics from each fold
        fold_accuracies = []
        fold_fit_times = []
        fold_fit_memories = []
        fold_predict_times = []
        fold_predict_memories = []

        fold_count = 0

        # 4. Manually loop through CV folds to control metric collection
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Define the preprocessing pipeline for this fold
            preprocessor = Pipeline([
                ('scaler', StandardScaler()),
                ('variance_filter', VarianceThreshold(threshold=0.1)),
                ('pca', PCA(n_components=0.9)) # Keep 90% of variance
            ])

            # Instantiate the classifier for this fold
            classifier = BaselineKNNClassifier()

            fold_count += 1
            print(f"Processing fold {fold_count}/{n_splits} for dataset '{dataset_name}'...")

            # Fit the preprocessing pipeline and transform the data
            # NOTE: We fit the preprocessor only on the training data
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            print(f"Fold {fold_count} - Preprocessing complete for dataset '{dataset_name}'")

            # --- Measure Fit Time and Memory ---
            tracemalloc.start()
            start_time = time.process_time()

            classifier.fit(X_train_processed, y_train)

            fit_time = time.process_time() - start_time
            _, fit_peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(f"Fold {fold_count} - Classifier fit complete for dataset '{dataset_name}'")

            # --- Measure Predict Time and Memory ---
            tracemalloc.start()
            start_time = time.process_time()
            
            y_pred = classifier.predict(X_test_processed)
            
            predict_time = time.process_time() - start_time
            _, predict_peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(f"Fold {fold_count} - Prediction complete for dataset '{dataset_name}'")

            # --- Calculate Accuracy ---
            accuracy = accuracy_score(y_test, y_pred)

            # --- Store metrics for this fold ---
            fold_accuracies.append(accuracy)
            fold_fit_times.append(fit_time)
            fold_fit_memories.append(fit_peak_mem / 1024**2) # Convert to MB
            fold_predict_times.append(predict_time)
            fold_predict_memories.append(predict_peak_mem / 1024**2) # Convert to MB

        # 5. Aggregate results after all folds are done
        results = {
            'dataset': dataset_name,
            'accuracy_mean': float(f"{np.mean(fold_accuracies):.3g}"),
            'accuracy_std': float(f"{np.std(fold_accuracies):.1g}"),
            'fit_time_cpu_mean': float(f"{np.mean(fold_fit_times):.3g}"),
            'fit_time_cpu_std': float(f"{np.std(fold_fit_times):.1g}"),
            'fit_mem_peak_mean_mb': float(f"{np.mean(fold_fit_memories):.3g}"),
            'fit_mem_peak_std_mb': float(f"{np.std(fold_fit_memories):.1g}"),
            'predict_time_cpu_mean': float(f"{np.mean(fold_predict_times):.3g}"),
            'predict_time_cpu_std': float(f"{np.std(fold_predict_times):.1g}"),
            'predict_mem_peak_mean_mb': float(f"{np.mean(fold_predict_memories):.3g}"),
            'predict_mem_peak_std_mb': float(f"{np.std(fold_predict_memories):.1g}")
        }
        print(f"✅ Finished processing for dataset: {dataset_name}")
        return results

    except Exception as e:
        print(f"❌ Error processing {dataset_name}: {e}")
        return None


def main():
    """Main function to parse arguments and run the parallel processing."""
    parser = argparse.ArgumentParser(description="Run cross-validation experiments in parallel.")
    parser.add_argument("datasets_folder", type=str, help="Path to the folder containing CSV datasets.")
    parser.add_argument("output_csv", type=str, help="Path to save the output results CSV file.")
    args = parser.parse_args()

    # Find all relevant dataset files
    input_path = Path(args.datasets_folder)
    output_path = Path(args.output_csv)
    
    if not input_path.is_dir():
        print(f"Error: Input path '{input_path}' is not a valid directory.")
        return

    # Get all .csv files that do not start with an underscore
    csv_files = [p for p in input_path.glob('*.csv') if not p.name.startswith('_')]

    if not csv_files:
        print(f"No valid CSV datasets found in '{input_path}'.")
        return

    print(f"Found {len(csv_files)} datasets to process.")
    
    # Use a process pool to run experiments in parallel
    # Uses all available CPU cores by default
    num_processes = cpu_count()
    print(f"Starting parallel execution with up to {num_processes} processes...")
    with Pool(processes=num_processes) as pool:
        # map() blocks until all results are ready
        results_list = pool.map(process_dataset, csv_files)

    # Filter out any `None` results from datasets that failed
    valid_results = [res for res in results_list if res is not None]

    if not valid_results:
        print("No datasets were successfully processed.")
        return
        
    # Convert results to a DataFrame and save to CSV
    results_df = pd.DataFrame(valid_results)
    
    # Reorder columns for clarity
    column_order = [
        'dataset', 'accuracy_mean', 'accuracy_std',
        'fit_time_cpu_mean', 'fit_time_cpu_std', 
        'predict_time_cpu_mean', 'predict_time_cpu_std',
        'fit_mem_peak_mean_mb', 'fit_mem_peak_std_mb',
        'predict_mem_peak_mean_mb', 'predict_mem_peak_std_mb'
    ]
    results_df = results_df[column_order]

    results_df.to_csv(output_path, index=False)
    print(f"\n🎉 All experiments complete! Results saved to '{output_path}'")


if __name__ == "__main__":
    main()