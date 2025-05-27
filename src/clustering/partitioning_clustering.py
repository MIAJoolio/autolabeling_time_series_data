from sklearn.cluster import KMeans
from tslearn import clustering
from sktime.clustering import k_medoids

from src.clustering.Base_clustering_model import Base_clustering_model


class Sklearn_kmeans_model(Base_clustering_model):
    def __init__(self):
        super().__init__()
        self.default_params = {
            "n_clusters": 8,
            "init": "k-means++",
            "n_init": 10,
            "max_iter": 300,
            "random_state": None
        }

    def fit_predict(self, X_train, X_test=None):
        X_train = self.scaler.normalize(X_train)
        
        model_params = self.default_params.copy()
        self.model = KMeans(**model_params)

        if X_test is not None:
            self.model.fit(X_train)
            labels = self.model.predict(X_test)
        else:
            labels = self.model.fit_predict(X_train)

        return labels, self.model


class Tslearn_kmeans_model(Base_clustering_model):
    def __init__(self):
        super().__init__()
        self.default_params = {
            "n_clusters": 3,
            "metric": "euclidean",
            "max_iter": 50,
            "random_state": None,
            "n_init": 10
        }

    def _reshape_data(self, data):
        """
        Преобразует данные в формат [n_samples, n_timesteps, n_features]
        """
        if len(data.shape) == 2:
            n_samples, n_timesteps = data.shape
            return data.reshape((n_samples, n_timesteps, 1))
        elif len(data.shape) == 3:
            return data
        else:
            raise ValueError("Data must be 2D or 3D")

    def fit_predict(self, X_train, X_test=None):
        model_params = self.default_params.copy()

        # Подготовка данных
        X_train = self._reshape_data(X_train)
        if X_test is not None:
            X_test = self._reshape_data(X_test)

        self.model = clustering.TimeSeriesKMeans(**model_params)
        self.model.fit(X_train)

        if X_test is not None:
            labels = self.model.predict(X_test)
        else:
            labels = self.model.predict(X_train)

        return labels, self.model


class Tslearn_kernel_model(Base_clustering_model):
    def __init__(self):
        super().__init__()
        self.default_params = {
            "n_clusters": 3,
            "kernel": "gak",
            "max_iter": 50,
            "n_init": 10,
            "random_state": None
        }

    def _reshape_data(self, data):
        if len(data.shape) == 2:
            return data.reshape((data.shape[0], data.shape[1], 1))
        elif len(data.shape) == 3:
            return data
        else:
            raise ValueError("Data must be 2D or 3D")

    def fit_predict(self, X_train, X_test=None):
        model_params = self.default_params.copy()

        X_train = self._reshape_data(X_train)
        if X_test is not None:
            X_test = self._reshape_data(X_test)

        self.model = clustering.KernelKMeans(**model_params)
        self.model.fit(X_train)

        if X_test is not None:
            labels = self.model.predict(X_test)
        else:
            labels = self.model.predict(X_train)

        return labels, self.model


class Sktime_kmedoids_model(Base_clustering_model):
    def __init__(self):
        super().__init__()
        self.default_params = {
            "n_clusters": 3,
            "metric": "euclidean",  # заменено с 'distance' на 'metric'
            "init_algorithm": "random",
            "max_iter": 100,
            "random_state": None
        }

    def _reshape_data(self, data):
        if len(data.shape) == 2:
            return data.reshape((data.shape[0], data.shape[1], 1))
        elif len(data.shape) == 3:
            return data
        else:
            raise ValueError("Data must be 2D or 3D")

    def fit_predict(self, X_train, X_test=None):
        model_params = self.default_params.copy()

        X_train = self._reshape_data(X_train)
        if X_test is not None:
            X_test = self._reshape_data(X_test)

        self.model = k_medoids.TimeSeriesKMedoids(**model_params)
        self.model.fit(X_train)

        if X_test is not None:
            labels = self.model.predict(X_test)
        else:
            labels = self.model.predict(X_train)

        return labels, self.model