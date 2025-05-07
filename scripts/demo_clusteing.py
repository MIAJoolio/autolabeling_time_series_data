import numpy as np
from sklearn.datasets import make_blobs

from src.clustering import DBSCAN_model, BIRCH_model, AC_model, Sklearn_kmeans_model, Tslearn_kmeans_model, Tslearn_kernel_model, Sktime_kmedoids_model


def density_check(X, y_true):
    """
    Проверяет работу плотностных методов кластеризации с автоматическим подбором параметров.
    """

    print("\n" + "=" * 60)
    print("Density-based Clustering: BIRCH")
    print("=" * 60)

    # Создаем модель BIRCH
    birch_model = BIRCH_model()

    # Загружаем YAML-конфигурацию и выполняем search_method
    df_birch_grid = birch_model.evaluate_results(X=X, y_true=y_true, config_path="configs/parameters_search/birch.yaml", search_method="grid_search")
    # print(df_birch[df_birch['metric'] == 'silhouette'].sort_values(by='value', ascending=False).head(3))
    print(df_birch_grid)
        

    print("\n" + "=" * 60)
    print("Density-based Clustering: DBSCAN")
    print("=" * 60)

    # Создаем модель DBSCAN
    dbscan_model = DBSCAN_model()

    # Загружаем YAML-конфигурацию и выполняем search_method
    df_dbscan_grid = dbscan_model.evaluate_results(X=X, y_true=y_true, config_path="configs/parameters_search/dbscan.yaml", search_method="grid_search")
    # print(df_dbscan[df_dbscan['metric'] == 'silhouette'].sort_values(by='value', ascending=False).head(3))
    print(df_dbscan_grid)
        

def agglomerative_check(X, y_true):
    """
    Проверяет Hierarchical метод (Agglomerative Clustering) с автоматическим подбором гиперпараметров.
    """
    
    print("\n" + "=" * 60)
    print("Hierarchical Clustering: Agglomerative")
    print("=" * 60)

    # Создаем модель
    aggl_model = AC_model()

    # Автоматический подбор гиперпараметров
    df_aggl = aggl_model.evaluate_results(X=X, y_true=y_true, config_path="configs/parameters_search/agglomerative.yaml", search_method="grid_search")
    # print(df_aggl[df_aggl['metric'] == 'silhouette'].sort_values(by='value', ascending=False).head(3))
    print(df_aggl)

def partitioning_check(X, y_true):
    # Искусственные данные
    X, y_true = make_blobs(n_samples=100, centers=3, random_state=42)
    
    model_objs = [Sklearn_kmeans_model(),  Sktime_kmedoids_model()] 
    # удалил Tslearn_kernel_model() - долгий
    # удалил Tslearn_kmeans_model() - долгий
    model_names = ['sklearn_kmeans.yaml',   'sktime_kmedoids.yaml'] 
    # удалил 'tslearn_kernel.yaml' - долгий
    # 'tslearn_kmeans.yaml',
    for model_obj, model_name in zip(model_objs, model_names):
    
        print("\n" + "=" * 60)
        print(f"{model_obj.__class__.__name__}")
        print("=" * 60)
    
        # Автоматический подбор гиперпараметров
        df_res = model_obj.evaluate_results(X=X, y_true=y_true, config_path=f"configs/parameters_search/{model_name}", search_method="grid_search")
        print(df_res)

def main():
    
    X_true, y_true = make_blobs(n_samples=100, centers=3, random_state=42)
    print(X_true.shape)
    
    density_check(X_true, y_true)
    agglomerative_check(X_true, y_true)
    partitioning_check(X_true, y_true)
    
if __name__ == '__main__':
    main()