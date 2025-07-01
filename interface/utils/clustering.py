import mlflow
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.signal import detrend
from statsmodels.tsa.seasonal import STL

def create_experiment(experiment_name, param_list, fe_method, X):
    mlflow.set_experiment(experiment_name)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    for idx, params in enumerate(param_list):
        with mlflow.start_run(experiment_id=experiment_id) as run:
            mlflow.log_params({f"fe_{k}": v for k, v in params.items()})

            if fe_method == 'dft':
                components = [np.fft.fft(row) for row in X]
                X_fe = np.array([np.abs(np.fft.ifft(c)) for c in components])
            elif fe_method == 'detrend':
                X_fe = np.array([detrend(row) for row in X])
            elif fe_method == 'stl':
                X_fe = []
                for row in X:
                    stl = STL(row, period=params['period'])
                    res = stl.fit()
                    X_fe.append(res.trend + res.seasonal)
                X_fe = np.array(X_fe)
            else:
                X_fe = X

            # Модель
            if 'kmeans' in experiment_name:
                model = KMeans(n_clusters=params.get('n_clusters', 2))
            elif 'agglomerative' in experiment_name:
                model = AgglomerativeClustering(
                    n_clusters=params.get('n_clusters', 2),
                    distance_threshold=params.get('distance_threshold', 0.5),
                    metric=params.get('metric', 'euclidean'),
                    linkage='average'
                )
            elif 'dbscan' in experiment_name:
                model = DBSCAN(
                    eps=params.get('eps', 0.5),
                    min_samples=params.get('min_samples', 5),
                    metric=params.get('metric', 'euclidean')
                )
            else:
                continue

            labels = model.fit_predict(X_fe)

            score = silhouette_score(X_fe, labels) if len(set(labels)) > 1 else 0
            mlflow.log_metric("silhouette_score", score)

            plt.figure(figsize=(8, 4))
            for i in range(min(5, len(X))):
                plt.plot(X[i], alpha=0.5)
                plt.plot(X_fe[i], '--')
            img_path = f"static/images/exp_{idx}.png"
            plt.savefig(img_path)
            plt.close()