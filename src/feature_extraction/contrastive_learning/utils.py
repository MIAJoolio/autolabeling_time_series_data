from typing import Literal, List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils import setup_logger, Logger 

def train_autoencoder_contrastive(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 10,
    model_save_path: Union[str, Path] = 'scripts/training_autoencoder',
    device: str = 'cuda:1',
    experiment_name: str = "contrastive_autoencoder_experiment",
    loss_function: str = "infonce"  # Добавлен параметр для выбора функции потерь
) -> Tuple[nn.Module, dict]:
    """
    Обучение автоэнкодера с контрастным обучением и логированием.
    
    Args:
        model: Модель автоэнкодера
        train_loader: DataLoader для обучающих данных
        val_loader: DataLoader для валидационных данных
        epochs: Максимальное количество эпох
        lr: Скорость обучения
        weight_decay: Коэффициент L2-регуляризации
        patience: Количество эпох для ранней остановки
        model_save_path: Путь для сохранения модели и логов
        device: Устройство для обучения
        experiment_name: Имя эксперимента для логирования
        loss_function: Выбор функции потерь ("infonce", "nt_xent", "triplet")
    
    Returns:
        Обученная модель и история обучения
    """
    # Инициализация логгера
    logger = setup_logger(experiment_name, level='debug')
    
    with logger.start_run(run_name=experiment_name):
        # Логирование параметров обучения
        logger.log_dict({
            "training": {
                "epochs": epochs,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "patience": patience,
                "device": device,
                "loss_function": loss_function  # Логируем выбранный тип функции потерь
            },
            "data": {
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "batch_size": train_loader.batch_size
            }
        })
        
        # Подготовка путей сохранения
        model_save_path = Path(model_save_path)
        model_save_path.mkdir(parents=True, exist_ok=True)
        
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        model = model.to(device).float()
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience//2)
        
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        best_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        logger.info("Starting training process...")
        
        for epoch in range(epochs):
            # Обучение
            model.train()
            train_loss = 0.0
            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.to(device).float()
                
                if loss_function in ["infonce", "nt_xent"]:
                    # Генерация положительных пар через маскирование
                    x1 = mask_augmentation(x, mask_ratio=0.2)
                    x2 = mask_augmentation(x, mask_ratio=0.2)
                    
                    # Получение латентных представлений
                    z1 = model(x1)
                    z2 = model(x2)
                    
                    if loss_function == "infonce":
                        # Генерация отрицательных пар (например, случайные объекты из батча)
                        negatives = torch.randn_like(x)
                        zn = model(negatives)
                        loss = model.infonce_loss(z1, z2, zn)
                    elif loss_function == "nt_xent":
                        loss = model.nt_xent_loss(z1, z2, temperature=model.temperature)
                
                elif loss_function == "triplet":
                    # Формирование троек (anchor, positive, negative)
                    anchor = x
                    positive = mask_augmentation(x, mask_ratio=0.2)
                    negative = torch.randn_like(x)
                    
                    # Получение латентных представлений
                    z_anchor = model(anchor)
                    z_positive = model(positive)
                    z_negative = model(negative)
                    
                    # Вычисление Triplet Loss
                    loss = model.triplet_loss(z_anchor, z_positive, z_negative, margin=1.0)
                
                else:
                    raise ValueError(f"Unsupported loss function: {loss_function}")
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * x.size(0)
                
                # Логирование каждые N батчей
                if batch_idx % 50 == 0:
                    logger.debug(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")
            
            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)
            logger.log_metric("train_loss", train_loss, step=epoch+1)
            
            # Валидация
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, _ in val_loader:
                    x = x.to(device).float()
                    
                    if loss_function in ["infonce", "nt_xent"]:
                        x1 = mask_augmentation(x, mask_ratio=0.2)
                        x2 = mask_augmentation(x, mask_ratio=0.2)
                        
                        z1 = model(x1)
                        z2 = model(x2)
                        
                        if loss_function == "infonce":
                            negatives = torch.randn_like(x)
                            zn = model(negatives)
                            loss = model.infonce_loss(z1, z2, zn)
                        elif loss_function == "nt_xent":
                            loss = model.nt_xent_loss(z1, z2, temperature=model.temperature)
                    
                    elif loss_function == "triplet":
                        anchor = x
                        positive = mask_augmentation(x, mask_ratio=0.2)
                        negative = torch.randn_like(x)
                        
                        z_anchor = model(anchor)
                        z_positive = model(positive)
                        z_negative = model(negative)
                        
                        loss = model.triplet_loss(z_anchor, z_positive, z_negative, margin=1.0)
                    
                    val_loss += loss.item() * x.size(0)
            
            val_loss /= len(val_loader.dataset)
            history['val_loss'].append(val_loss)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            logger.log_metric("val_loss", val_loss, step=epoch+1)
            logger.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch+1)
            
            # Обновление learning rate
            scheduler.step(val_loss)
            
            # Ранняя остановка
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), model_save_path/'best_model.pth')
                logger.info(f"New best model at epoch {epoch+1} with val loss {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f'Early stopping triggered at epoch {epoch+1}')
                    break
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
        
        # Сохранение результатов
        pd.DataFrame(history).to_csv(model_save_path/'training_history.csv', index=False)
        logger.save_metrics()
        logger.save_params()
        
        # Загрузка лучшей модели
        model.load_state_dict(torch.load(model_save_path/'best_model.pth'))
        logger.info(
            f"Training completed. Best model at epoch {best_epoch+1} "
            f"with val loss {best_loss:.4f}"
        )
        
        return model, history

def mask_augmentation(x: torch.Tensor, mask_ratio: float = 0.2) -> torch.Tensor:
    """
    Применяет маскирование к входным данным.
    
    Args:
        x: Входной тензор размерности [batch_size, input_dim].
        mask_ratio: Доля элементов, которые будут замаскированы.
    
    Returns:
        torch.Tensor: Маскированная версия входных данных.
    """
    batch_size, input_dim = x.size()
    
    # Генерация маски
    mask = torch.rand_like(x) > mask_ratio  # True для элементов, которые остаются нетронутыми
    
    # Применение маски
    masked_x = x * mask.float()  # Заменяем маскированные элементы на 0
    
    return masked_x
