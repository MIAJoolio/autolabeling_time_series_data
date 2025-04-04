import torch
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from core.generation.ts_datasets import Synthetic_dataset
from core.feature_extraction.scipy_signal import (
    scipy_trend, scipy_seasonality, scipy_structural_changes,
    scipy_noise, scipy_cross_correlation
)
from core.feature_extraction.other_methods import (
    test_method_statistics, test_method_peaks, test_method_stft,
    test_method_dft, test_method_dwt, test_method_paa
)
from core.clustering.partitioning_clustering import kmeans_clustering
from core.clustering.hierarchical_clustering import hierarchical_clustering
from core.clustering.density_clustering import dbscan_clustering

def setup_logging():
    """Настройка логирования"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'comparison_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def extract_features(series, get_features=True):
    """Извлечение признаков всеми методами"""
    features = {}
    
    # Scipy Signal методы
    features['trend'] = scipy_trend(series, get_features=get_features)
    features['seasonality'] = scipy_seasonality(series, get_features=get_features)
    features['structural'] = scipy_structural_changes(series, get_features=get_features)
    features['noise'] = scipy_noise(series, get_features=get_features)
    
    # Other methods
    features['statistics'] = test_method_statistics(series, get_features=get_features)
    features['peaks'] = test_method_peaks(series, get_features=get_features)
    features['stft'] = test_method_stft(series, get_features=get_features)
    features['dft'] = test_method_dft(series, get_features=get_features)
    features['dwt'] = test_method_dwt(series, get_features=get_features)
    features['paa'] = test_method_paa(series, get_features=get_features)
    
    return features

def get_latent_representations(model, dataloader, device):
    """Получение латентных представлений из модели"""
    model.eval()
    latent_vectors = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            latent = model.encode(x)
            latent_vectors.append(latent.cpu().numpy())
    return np.vstack(latent_vectors)

def evaluate_clustering(X, labels, logger):
    """Оценка качества кластеризации"""
    try:
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        davies = davies_bouldin_score(X, labels)
        
        logger.info(f'Silhouette Score: {silhouette:.4f}')
        logger.info(f'Calinski-Harabasz Score: {calinski:.4f}')
        logger.info(f'Davies-Bouldin Score: {davies:.4f}')
        
        return {
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies
        }
    except Exception as e:
        logger.error(f'Ошибка при оценке кластеризации: {str(e)}')
        return None

def main():
    # Настройка логирования
    logger = setup_logging()
    logger.info('Начало сравнения методов')
    
    # Загрузка данных
    logger.info('Загрузка данных...')
    dataset = Synthetic_dataset('data/4block_dataset.json')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    
    # Загрузка модели
    logger.info('Загрузка модели...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('models/best_model.pth', map_location=device)
    
    # Получение латентных представлений
    logger.info('Получение латентных представлений...')
    latent_vectors = get_latent_representations(model, dataloader, device)
    
    # Извлечение признаков
    logger.info('Извлечение признаков...')
    all_features = []
    for series in dataset.series:
        features = extract_features(series, get_features=True)
        feature_vector = np.concatenate([
            np.array(list(features['trend'].values())),
            np.array(list(features['seasonality'].values())),
            np.array(list(features['structural'].values())),
            np.array(list(features['noise'].values())),
            np.array(list(features['statistics'].values())),
            np.array(list(features['peaks'].values())),
            np.array(list(features['stft'].values())),
            np.array(list(features['dft'].values())),
            np.array(list(features['dwt'].values())),
            np.array(list(features['paa'].values()))
        ])
        all_features.append(feature_vector)
    
    all_features = np.array(all_features)
    
    # Нормализация данных
    scaler = StandardScaler()
    latent_vectors_scaled = scaler.fit_transform(latent_vectors)
    all_features_scaled = scaler.fit_transform(all_features)
    
    # Кластеризация и оценка
    results = {}
    
    # K-means
    logger.info('Применение K-means...')
    kmeans_labels = kmeans_clustering(latent_vectors_scaled, n_clusters=8)
    results['kmeans_latent'] = evaluate_clustering(latent_vectors_scaled, kmeans_labels, logger)
    
    kmeans_labels = kmeans_clustering(all_features_scaled, n_clusters=8)
    results['kmeans_features'] = evaluate_clustering(all_features_scaled, kmeans_labels, logger)
    
    # Иерархическая кластеризация
    logger.info('Применение иерархической кластеризации...')
    hierarchical_labels = hierarchical_clustering(latent_vectors_scaled, n_clusters=8)
    results['hierarchical_latent'] = evaluate_clustering(latent_vectors_scaled, hierarchical_labels, logger)
    
    hierarchical_labels = hierarchical_clustering(all_features_scaled, n_clusters=8)
    results['hierarchical_features'] = evaluate_clustering(all_features_scaled, hierarchical_labels, logger)
    
    # DBSCAN
    logger.info('Применение DBSCAN...')
    dbscan_labels = dbscan_clustering(latent_vectors_scaled, eps=0.5, min_samples=5)
    results['dbscan_latent'] = evaluate_clustering(latent_vectors_scaled, dbscan_labels, logger)
    
    dbscan_labels = dbscan_clustering(all_features_scaled, eps=0.5, min_samples=5)
    results['dbscan_features'] = evaluate_clustering(all_features_scaled, dbscan_labels, logger)
    
    # Сохранение результатов
    with open('results/clustering_comparison.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info('Сравнение методов завершено!')

if __name__ == '__main__':
    main() 