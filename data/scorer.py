from sklearn import metrics
import pandas as pd
import numpy as np  # Import numpy

def evaluate_clustering_result(y_true, y_pred, X=None):
    return {
        'ARI': metrics.adjusted_rand_score(y_true, y_pred),
        'AMI': metrics.adjusted_mutual_info_score(y_true, y_pred),
        'Homogeneity': metrics.homogeneity_score(y_true, y_pred),
        'Completeness': metrics.completeness_score(y_true, y_pred),
        'V-measure': metrics.v_measure_score(y_true, y_pred),
        'Silhouette': metrics.silhouette_score(X, y_pred) if X is not None and len(np.unique(y_pred)) > 1 else None
        }

def compare_clustering_results(labels_dict):
    """
    Compare clustering results using various evaluation metrics.

    Parameters:
        y_true (np.ndarray): True labels of shape (n_samples,).
        labels_dict (dict): A dictionary where keys are algorithm names and values are predicted labels.
        X (np.ndarray): Input data of shape (n_samples, n_features). Required for Silhouette Score.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation metrics for each algorithm.
    """
    # Evaluate each algorithm
    data = []
    for name, params_cond in labels_dict.items():
        for k, v in params_cond.items():
            y_pred, y_true, X, params = v.values()
            result_dict = evaluate_clustering_result(y_true, y_pred, X)
            result_dict['initial_condition'] = name
            result_dict['params'] = params
            result_dict['params_key'] = k
            data.append(result_dict)

    # Create a DataFrame to display the results
    results = pd.DataFrame(data)
    # results.set_index('initial_condition', inplace=True)
    return results