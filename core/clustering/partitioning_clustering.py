import numpy as np

from sklearn.cluster import KMeans
from sktime.clustering import k_means 
from tslearn import clustering

def apply_kmeans(X_train, X_test=None, **model_params):
    """
    Apply K-Means clustering to the data.

    Parameters:
        X (np.ndarray): 2D array of shape (n_samples, n_timesteps) for time-series or (n_samples, n_features) for points.
        model_params: any parameters from sklearn.clustering https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    
    Returns:
        np.ndarray: Cluster labels.
    """
    # Apply K-Means
    model = KMeans(**model_params)

    if X_test is not None:
        model.fit(X_train)
        return model.predict(X_test), model
    
    return model.fit_predict(X_train), model

def apply_ts_kmeans(X_train, X_test=None, **model_params):
    """
    Apply TimeSeriesKMeans clustering to the data.

    Parameters:
        X_train (np.ndarray): 2D array of shape (n_samples, n_timesteps) for time-series data.
        X_test (np.ndarray, optional): 2D array of shape (n_samples, n_timesteps) for time-series data to predict clusters.
                                      If None, only training is performed.
        model_params: Parameters for TimeSeriesKMeans, such as `n_clusters`, `metric`, `max_iter`, etc.
                      See https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.TimeSeriesKMeans.html

    Returns:
        np.ndarray: Cluster labels for the input data.
        TimeSeriesKMeans: Fitted model.
    """
    if len(X_train.shape) == 2:
        
        for data in [data for data in [X_train, X_test] if data is not None]:
            # количество временных рядов
            n_samples = data.shape[0]
            # количество временных точек в каждом ряду
            n_timesteps = data.shape[1]
            # n_features количество признаков (измерений) в каждой временной точке 
            n_features = 1

            data = data.reshape(n_samples, n_timesteps, n_features)

    elif len(data.shape) == 3:
        print("Корректный формат данных, но проверьте, что data.shape = [количество временных рядов, количество временных точек в каждом ряду, количество признаков (измерений) в каждой временной точке]") 

    else:
        raise ValueError('Входные данные должны быть в формате [количество временных рядов, количество временных точек в каждом ряду, количество признаков (измерений) в каждой временной точке] или для одномерных массивов [количество временных рядов, количество временных точек в каждом ряду]')

    # Initialize the TimeSeriesKMeans model with provided parameters
    model = clustering.TimeSeriesKMeans(**model_params)

    # Fit the model on the training data
    model.fit(X_train)

    # Predict clusters for the test data if provided
    if X_test is not None:
        labels = model.predict(X_test)
        return labels, model

    # If no test data, return labels for the training data
    labels = model.predict(X_train)
    return labels, model


def apply_ts_kernel_kmeans(X_train, X_test=None, **model_params):
    """
    Apply KernelKMeans clustering to the data.

    Parameters:
        X_train (np.ndarray): 2D or 3D array of time-series data.
        X_test (np.ndarray, optional): 2D or 3D array of time-series data to predict clusters.
        model_params: Parameters for KernelKMeans.

    Returns:
        np.ndarray: Cluster labels.
        KernelKMeans: Fitted model.
    """
    if X_train.ndim not in [2, 3]:
        raise ValueError("Input data must be 2D or 3D.")

    if X_train.ndim == 2:
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    if X_test is not None:
        if X_test.ndim not in [2, 3]:
            raise ValueError("Test data must be 2D or 3D.")
        if X_test.ndim == 2:
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = clustering.KernelKMeans(**model_params)

    if X_test is not None:
        model.fit(X_train)
        return model.predict(X_test), model

    return model.fit_predict(X_train), model


def apply_sk_kmeans(X_train, X_test, **model_params):
    """
    Apply K-Means clustering to the data.

    Parameters:
        X (np.ndarray): 2D array of shape (n_samples, n_timesteps) for time-series or (n_samples, n_features) for points.
        model_params: any parameters from tslearn.clustering https://tslearn.readthedocs.io/en/latest/gen_modules/clustering/tslearn.clustering.TimeSeriesKMeans.html#tslearn.clustering.TimeSeriesKMeans 
    
    Returns:
        np.ndarray: Cluster labels.
    """
    # Apply K-Means
    model = k_means.TimeSeriesKMeans(**model_params)
    
    if X_test is not None:
        model.fit(X_train)
        return model.predict(X_test), model

    return model.fit_predict(X_train), model


def apply_sk_kmedoids(X_train, X_test, **model_params):
    """
    Apply K-Means clustering to the data.

    Parameters:
        X (np.ndarray): 2D array of shape (n_samples, n_timesteps) for time-series or (n_samples, n_features) for points.
        model_params: any parameters from tslearn.clustering https://tslearn.readthedocs.io/en/latest/gen_modules/clustering/tslearn.clustering.TimeSeriesKMeans.html#tslearn.clustering.TimeSeriesKMeans 
    
    Returns:
        np.ndarray: Cluster labels.
    """
    # Apply K-Means
    model = k_means.TimeSeriesKMedoids(**model_params)
    
    if X_test is not None:
        model.fit(X_train)
        return model.predict(X_test), model

    return model.fit_predict(X_train), model


def main():
    return None

if __name__ == '__main__':
    main()
