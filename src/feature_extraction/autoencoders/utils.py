from typing import Literal, List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.utils import setup_logger, Logger, plot_series_grid
from src.feature_extraction.autoencoders.models import Basic_autoencoder


__all__ = [
    'Training_config',
    'Basic_trainer',
    'extract_latent_features',
    'visualize_all_latent_points'
]

@dataclass
class Training_config:
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    model_save_path: Union[str, Path] = "scripts/experiment1"
    device: str = "cuda:1"
    experiment_name: str = "autoencoder_experiment1"
    log_dir: str = ".logs"


class Basic_trainer:
    def __init__(
        self,
        model: Basic_autoencoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Training_config,
        logger: Optional[Logger] = None
    ):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).float()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger or setup_logger(
            name=config.experiment_name,
            log_dir=config.log_dir,
            level='debug'
        )
        self.criterion = self._get_criterion()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.model_save_path = self._prepare_model_save_path()
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0

    def _get_criterion(self) -> nn.Module:
        return nn.MSELoss()

    def _get_optimizer(self) -> optim.Optimizer:
        return optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

    def _get_scheduler(self) -> ReduceLROnPlateau:
        return ReduceLROnPlateau(self.optimizer, mode='min', patience=self.config.patience // 2)

    def _prepare_model_save_path(self) -> Path:
        path = Path(self.config.model_save_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _log_training_params(self):
        self.logger.log_dict({
            "training": {
                "epochs": self.config.epochs,
                "learning_rate": self.config.lr,
                "weight_decay": self.config.weight_decay,
                "patience": self.config.patience,
                "device": str(self.device),
            },
            "data": {
                "train_samples": len(self.train_loader.dataset),
                "val_samples": len(self.val_loader.dataset),
                "batch_size": self.train_loader.batch_size
            }
        })

    def compute_loss(self, x, reconstructed, *args, **kwargs):
        if len(x.shape) > 2:
            x = x.view_as(reconstructed)
        return self.criterion(reconstructed, x)

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch_idx, (x, _) in enumerate(self.train_loader):
            x = x.to(self.device).float()
            self.optimizer.zero_grad()
            reconstructed, _ = self.model(x)
            loss = self.compute_loss(x, reconstructed)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
            if batch_idx % 50 == 0:
                self.logger.debug(f"Batch {batch_idx} | Loss: {loss.item():.4f}")
        return total_loss / len(self.train_loader.dataset)

    def _validate_one_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(self.val_loader):
                x = x.to(self.device).float()
                reconstructed, _ = self.model(x)
                loss = self.compute_loss(x, reconstructed)
                total_loss += loss.item() * x.size(0)
        return total_loss / len(self.val_loader.dataset)

    def _handle_early_stopping(self, val_loss: float) -> bool:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = len(self.history['val_loss'])
            self.patience_counter = 0
            torch.save(self.model.state_dict(), self.model_save_path / 'best_model.pth')
            self.logger.info(f"New best model at epoch {self.best_epoch + 1} with val loss {val_loss:.4f}")
        else:
            self.patience_counter += 1
        return self.patience_counter >= self.config.patience

    def _save_training_results(self):
        pd.DataFrame(self.history).to_csv(self.model_save_path / 'training_history.csv', index=False)
        self.logger.save_metrics()
        self.logger.save_params()
        self.model.load_state_dict(torch.load(self.model_save_path / 'best_model.pth'))
        self.logger.info(
            f"Training completed. Best model at epoch {self.best_epoch + 1} "
            f"with val loss {self.best_val_loss:.4f}"
        )

    def _plot_training_loss(self):
        plot_series_grid(
            series_list=[self.history["train_loss"], self.history["val_loss"]],
            labels=["Train Loss", "Validation Loss"],
            x_series=np.arange(1, len(self.history["train_loss"]) + 1),
            plot_title="Training and Validation Loss",
            ylabel="Loss (MSE)",
            xlabel="Epoch",
            figsize=(12, 6),
            grid=True,
            layout="vertical",
            save_path=str(self.model_save_path / 'training_loss_plot.png')
        )

    def train(self) -> Tuple[nn.Module, dict]:
        with self.logger.start_run(run_name=self.config.experiment_name):
            self._log_training_params()
            self.logger.info("Starting training process...")
            for epoch in range(self.config.epochs):
                train_loss = self._train_one_epoch()
                val_loss = self._validate_one_epoch()
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['lr'].append(current_lr)
                self.scheduler.step(val_loss)
                early_stop = self._handle_early_stopping(val_loss)
                self.logger.log_metric("train_loss", train_loss, step=epoch + 1)
                self.logger.log_metric("val_loss", val_loss, step=epoch + 1)
                self.logger.log_metric("learning_rate", current_lr, step=epoch + 1)
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"LR: {current_lr:.2e}"
                )
                if early_stop:
                    self.logger.info(f'Early stopping triggered at epoch {epoch + 1}')
                    break
            self._save_training_results()
            self._plot_training_loss()
        return self.model, self.history


def extract_latent_features(
    model: Basic_autoencoder,
    data_loader: DataLoader,
    device: str = 'cuda:1',
    logger: Optional[Logger] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Извлечение латентных представлений с логированием.
    Работает с любым объектом, наследующим AutoencoderBase.
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
            # Автоматически обрабатываем данные через forward
            _, latent = model(x)
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

 
def visualize_all_latent_points(
    latents: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    title: str = "Latent Space Visualization"
):
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents) - 1))
    latent_2d = tsne.fit_transform(latents)

    unique_labels = np.unique(labels)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    mapped_labels = np.array([label_map[label] for label in labels])

    cmap = plt.cm.get_cmap('tab10', len(unique_labels))

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=mapped_labels, cmap=cmap, alpha=0.7, s=50)

    norm = BoundaryNorm(np.arange(len(unique_labels) + 1) - 0.5, ncolors=len(unique_labels))
    cbar = plt.colorbar(scatter, ticks=np.arange(len(unique_labels)), norm=norm)
    cbar.set_ticklabels(unique_labels)
    cbar.set_label("Class Label")

    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    

def visualize_reconstructions_by_class(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    class_names: Dict[int, str],
    num_samples_per_class: int = 3,
    device: str = 'cpu',
    images_save_path: Union[str, Path] = None
):
    """
    Визуализирует реконструкции временных рядов по классам.
    
    :param model: обученная модель LSTM_autoencoder
    :param data_loader: DataLoader, содержащий данные с метками (y)
    :param class_names: словарь {class_id: class_name}
    :param num_samples_per_class: количество примеров на класс для отображения
    :param device: устройство ('cpu' или 'cuda')
    :param images_save_path: путь к папке для сохранения изображений
    """
    model.eval()
    model.to(device)

    # Словарь для хранения данных по классам
    class_to_samples = {cls: [] for cls in class_names}

    with torch.no_grad():
        # Проходим по данным до заполнения всех классов
        for batch in data_loader:
            x_batch, y_batch = batch[0].to(device).float(), batch[1].to(device).long()

            for i in range(len(y_batch)):
                label = y_batch[i].item()
                if len(class_to_samples[label]) < num_samples_per_class:
                    x_sample = x_batch[i:i+1]  # (1, seq_len, input_size)
                    recon_sample, _ = model(x_sample)
                    class_to_samples[label].append((x_sample.cpu().numpy()[0],
                                                    recon_sample.cpu().numpy()[0]))

            # Проверяем, собраны ли все нужные образцы
            all_full = all(len(samples) >= num_samples_per_class for samples in class_to_samples.values())
            if all_full:
                break

    # Создаём папку для сохранения, если указана
    if images_save_path is not None:
        save_dir = Path(images_save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Для каждого класса строим графики
    for cls, samples in class_to_samples.items():
        class_name = class_names[cls]
        print(f"Plotting reconstructions for class '{class_name}' ({cls})")

        for idx, (original, reconstructed) in enumerate(samples):
            plt.figure(figsize=(10, 3))
            plt.plot(original[:, 0], label='Original', color='blue', linewidth=2)
            plt.plot(original[:, 0], label='Reconstructed', color='red', linestyle='--', linewidth=2)
            plt.title(f"{class_name} - Sample {idx + 1}")
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)

            if images_save_path is not None:
                save_path = Path(images_save_path) / f"{class_name.replace(' ', '_')}_sample_{idx + 1}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved plot to {save_path}")

            plt.show()

