import numpy as np

from sklearn.cluster import AgglomerativeClustering

def apply_agglomerative(X_train, X_test=None, **model_params):
    """
    Apply Agglomerative Clustering to the data.

    Parameters:
        X_train (np.ndarray): 2D array of shape (n_samples, n_timesteps) for time-series or (n_samples, n_features) for points.
        X_test (np.ndarray): 2D array of shape (n_samples, n_timesteps) for time-series or (n_samples, n_features) for points.
        model_params: any parameters from sklearn.cluster.AgglomerativeClustering
    
    Returns:
        np.ndarray: Cluster labels.
    """
    # Apply Agglomerative Clustering
    model = AgglomerativeClustering(**model_params)
    
    if X_test is not None:
        model.fit(X_train)
        return model.fit_predict(X_test), model
    
    return model.fit_predict(X_train), model

# чет можно еще для одного ряда побаловаться - https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering.html#sphx-glr-auto-examples-cluster-plot-agglomerative-clustering-py 



def main():
    return None

if __name__ == '__main__':
    main()