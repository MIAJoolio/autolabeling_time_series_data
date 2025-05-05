from typing import Literal, List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils import setup_logger, Logger, plot_series_grid


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 10,
    model_save_path: Union[str, Path] = 'scripts/training_autoencoder',
    device: str = 'cuda:1',
    experiment_name: str = "autoencoder_experiment",
    model_type: Literal["fc" , "lstm"] = "fc"
) -> Tuple[nn.Module, dict]:
    """
    Обучение автоэнкодера с логированием и ранней остановкой.
    """
    
    def _log_training_params(
        logger: Logger,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float,
        weight_decay: float,
        patience: int,
        device: str,
        model_type: str):
        
        logger.log_dict({
            "training": {
                "epochs": epochs,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "patience": patience,
                "device": device,
                "model_type": model_type
            },
            "data": {
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "batch_size": train_loader.batch_size
            }
        })
        
    def _prepare_model_save_path(model_save_path: Union[str, Path]) -> Path:
        model_save_path = Path(model_save_path)
        model_save_path.mkdir(parents=True, exist_ok=True)
        return model_save_path

    def _train_one_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        model_type: str,
        logger: Logger,
        epoch: int,
        total_epochs: int ) -> float:
        
        model.train()
        train_loss = 0.0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device).float()
            if model_type == "fc":
                x = model.flatten_data(x) 
            optimizer.zero_grad()
            reconstructed, _ = model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            if batch_idx % 50 == 0:
                logger.debug(f"Epoch {epoch + 1}/{total_epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        train_loss /= len(train_loader.dataset)
        return train_loss
    
    def _validate_one_epoch(
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        model_type: str,
        logger: Logger) -> float:
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device).float()
                if model_type == "fc":
                    x = model.flatten_data(x)  
                reconstructed, _ = model(x)
                loss = criterion(reconstructed, x)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        return val_loss

    def _handle_early_stopping(
        val_loss: float,
        best_loss: float,
        best_epoch: int,
        patience_counter: int,
        patience: int,
        epoch: int,
        model: nn.Module,
        model_save_path: Path,
        logger: Logger) -> Tuple[float, int, int]:
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path / 'best_model.pth')
            logger.info(f"New best model at epoch {epoch + 1} with val loss {val_loss:.4f}")
        else:
            patience_counter += 1
        return best_loss, best_epoch, patience_counter

    def _save_training_results(
        history: dict,
        model_save_path: Path,
        model: nn.Module,
        logger: Logger):
        
        pd.DataFrame(history).to_csv(model_save_path / 'training_history.csv', index=False)
        logger.save_metrics()
        logger.save_params()
        model.load_state_dict(torch.load(model_save_path / 'best_model.pth'))
        logger.info(
            f"Training completed. Best model at epoch {history['val_loss'].index(min(history['val_loss'])) + 1} "
            f"with val loss {min(history['val_loss']):.4f}"
        )

    
    def _plot_training_loss(
        history: dict,
        save_path: Optional[str] = None,
        plot_title: str = "Training and Validation Loss",
        ylabel: str = "Loss (MSE)",
        xlabel: str = "Epoch",
        figsize: tuple = (12, 6),
        layout: str = "vertical"):
        """
        Визуализация MSE train loss и MSE test loss из словаря history.
        
        Args:
            history (dict): Словарь с историей обучения (ключи: 'train_loss', 'val_loss').
            save_path (Optional[str]): Путь для сохранения графика. Если None, график только отображается.
            plot_title (str): Заголовок графика.
            ylabel (str): Подпись оси Y.
            xlabel (str): Подпись оси X.
            figsize (tuple): Размер фигуры (ширина, высота).
            layout (str): Расположение графиков ('vertical', 'horizontal', 'grid').
        """
        # Извлечение значений train_loss и val_loss
        train_loss = history["train_loss"]
        val_loss = history["val_loss"]
        epochs = np.arange(1, len(train_loss) + 1)

        # Вызов функции plot_series_grid
        plot_series_grid(
            series_list=[train_loss, val_loss],
            labels=["Train Loss", "Validation Loss"],
            x_series=epochs,
            plot_title=plot_title,
            ylabel=ylabel,
            xlabel=xlabel,
            figsize=figsize,
            grid=True,
            layout=layout,
            save_path=save_path
        )


    # Инициализация логгера
    logger = setup_logger(experiment_name, level='debug')
    with logger.start_run(run_name=experiment_name):
        # Логирование параметров обучения
        _log_training_params(logger, train_loader, val_loader, epochs, lr, weight_decay, patience, device, model_type)
        
        # Подготовка путей сохранения
        model_save_path = _prepare_model_save_path(model_save_path)
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        model = model.to(device).float()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience // 2)
        
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        best_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        logger.info("Starting training process...")
        for epoch in range(epochs):
            # Обучение
            train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device, model_type, logger, epoch, epochs)
            history['train_loss'].append(train_loss)
            logger.log_metric("train_loss", train_loss, step=epoch + 1)
            
            # Валидация
            val_loss = _validate_one_epoch(model, val_loader, criterion, device, model_type, logger)
            history['val_loss'].append(val_loss)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            logger.log_metric("val_loss", val_loss, step=epoch + 1)
            logger.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch + 1)
            
            # Обновление learning rate
            scheduler.step(val_loss)
            
            # Ранняя остановка
            best_loss, best_epoch, patience_counter = _handle_early_stopping(
                val_loss, best_loss, best_epoch, patience_counter, patience, epoch, model, model_save_path, logger
            )
            
            if patience_counter >= patience:
                logger.info(f'Early stopping triggered at epoch {epoch + 1}')
                break
            
            logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
        
        # Сохранение результатов
        _save_training_results(history, model_save_path, model, logger)
        _plot_training_loss(
                history=history,
                save_path=str(model_save_path / 'training_loss_plot.png'),
                plot_title="Training and Validation Loss",
                ylabel="Loss (MSE)",
                xlabel="Epoch",
                figsize=(12, 6),
                layout="vertical"
            )
        
        return model, history
    
    
def extract_latent_features(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cuda:1',
    logger: Optional[Logger] = None,
    model_type: Literal['fc', 'lstm'] = 'fc'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Извлечение латентных представлений с логированием.
    
    Args:
        model: Обученная модель автоэнкодера.
        data_loader: DataLoader с данными.
        device: Устройство для вычислений.
        logger: Экземпляр логгера (опционально).
        model_type: Тип модели ("fc" для полносвязного автоэнкодера, "lstm" для LSTM).
    
    Returns:
        Латентные представления и соответствующие метки.
    """
    if logger is None:
        logger = setup_logger("LatentExtractor", level='info')
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    latents = []
    labels = []
    logger.info("Starting latent features extraction...")
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            x = x.to(device).float()
            
            if model_type == "fc":
                # Для полносвязного автоэнкодера разворачиваем данные
                x = model.flatten_data(x)
                latent = model.encoder(x)
            
            elif model_type == "lstm":
                # Для LSTM берем последнее скрытое состояние
                _, (hidden, _) = model.encoder(x)  # LSTM возвращает (output, (hidden, cell))
                latent = hidden[-1]  # Берем последнее скрытое состояние последнего слоя
            
            latents.append(latent.cpu().numpy())
            labels.append(y.numpy())
            
            if batch_idx % 50 == 0:
                logger.debug(f"Processed {batch_idx * data_loader.batch_size} samples")
    
    latents = np.vstack(latents)
    labels = np.concatenate(labels)
    logger.info(
        f"Extraction completed. Got {latents.shape[0]} samples "
        f"with {latents.shape[1]} latent dimensions"
    )
    return latents, labels


def visualize_all_latent_points(latents: np.ndarray, labels: np.ndarray, save_path: str, title: str = "Latent Space Visualization"):
    """
    Визуализация всех латентных представлений с помощью t-SNE.
    
    Args:
        latents: Латентные представления (N_samples x latent_dim)
        labels: Метки классов (N_samples,)
        save_path: Путь для сохранения графика
        title: Заголовок графика
    """
    # Применяем t-SNE для снижения размерности до 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents) - 1))
    latent_2d = tsne.fit_transform(latents)

    # Преобразуем метки классов в последовательные целые числа
    unique_labels = np.unique(labels)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    mapped_labels = np.array([label_map[label] for label in labels])

    # Создаем дискретную цветовую карту
    num_classes = len(unique_labels)
    cmap = plt.cm.get_cmap('tab10', num_classes)  # Получаем дискретную цветовую карту

    # Создаем scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        latent_2d[:, 0], 
        latent_2d[:, 1], 
        c=mapped_labels, 
        cmap=cmap, 
        alpha=0.7, 
        s=50
    )

    # Добавляем colorbar с правильным количеством классов
    norm = BoundaryNorm(np.arange(num_classes + 1) - 0.5, ncolors=num_classes)
    cbar = plt.colorbar(scatter, ticks=np.arange(num_classes), norm=norm)
    cbar.set_ticklabels(unique_labels)  # Устанавливаем метки классов на colorbar
    cbar.set_label("Class Label")

    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True)
    plt.savefig(save_path)
    # plt.show()

