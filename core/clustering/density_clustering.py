from sklearn.cluster import DBSCAN, Birch

import numpy as np
import matplotlib.pyplot as plt

def apply_dbscan(X_train, X_test=None, **model_params):
    """
    Apply DBSCAN clustering to the data.

    Parameters:
        X_train (np.ndarray): 2D array of shape (n_samples, n_timesteps) for time-series or (n_samples, n_features) for points.
        X_test (np.ndarray): 2D array of shape (n_samples, n_timesteps) for time-series or (n_samples, n_features) for points.
        model_params: any parameters from sklearn.cluster.DBSCAN
    
    Returns:
        np.ndarray: Cluster labels.
    """
    # Apply DBSCAN
    model = DBSCAN(**model_params)
    
    if X_test is None:
        return model.fit_predict(X_train), model
    
    model.fit(X_train)
    return model.fit_predict(X_test), model


def apply_birch(X_train, X_test=None, **model_params):
    """
    Apply BIRCH clustering to the data.

    Parameters:
        X_train (np.ndarray): 2D array of shape (n_samples, n_timesteps) for time-series or (n_samples, n_features) for points.
        X_test (np.ndarray): 2D array of shape (n_samples, n_timesteps) for time-series or (n_samples, n_features) for points.
        model_params: any parameters from sklearn.cluster.Birch
    
    Returns:
        np.ndarray: Cluster labels.
    """
    # Apply BIRCH
    model = Birch(**model_params)
    
    if X_test is None:
        return model.fit_predict(X_train), model
    
    model.fit(X_train)
    return model.predict(X_test), model

def main():
    return None

if __name__ == '__main__':
    main()