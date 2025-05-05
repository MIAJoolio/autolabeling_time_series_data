from typing import Literal, List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import setup_logger, Logger 

class Contrastive_ae_two_layer(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32, dropout_rate: float = 0.2, temperature: float = 0.1):
        super(Contrastive_ae_two_layer, self).__init__()
        
        # Энкодер
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, latent_dim)
        )
        
        # Параметры контрастного обучения
        self.temperature = temperature
        
        # Логирование параметров модели
        self.logger = setup_logger(f"{self.__class__.__name__}", level='debug')
        self.logger.info(f"Initializing Autoencoder with input_dim={input_dim}, latent_dim={latent_dim}")
        self.logger.log_param("input_dim", input_dim)
        self.logger.log_param("latent_dim", latent_dim)
        self.logger.log_param("dropout_rate", dropout_rate)
        self.logger.log_param("temperature", temperature)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через энкодер.
        
        Args:
            x: Входной тензор размерности [batch_size, input_dim].
        
        Returns:
            torch.Tensor: Латентные представления размерности [batch_size, latent_dim].
        """
        x = x.float()
        latent = self.encoder(x)
        return latent
    
    def infonce_loss(self, z1: torch.Tensor, z2: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет контрастную функцию потерь (InfoNCE Loss).
        
        Args:
            z1: Латентные представления первого объекта (положительная пара).
            z2: Латентные представления второго объекта (положительная пара).
            negatives: Латентные представления отрицательных объектов.
        
        Returns:
            torch.Tensor: Значение функции потерь.
        """
        batch_size = z1.size(0)
        
        # Вычисляем косинусное подобие между всеми парами
        sim_pos = F.cosine_similarity(z1, z2, dim=-1) / self.temperature
        sim_neg = torch.matmul(z1, negatives.T) / self.temperature
        
        # Вычисляем числитель и знаменатель InfoNCE Loss
        numerator = torch.exp(sim_pos)
        denominator = numerator + torch.sum(torch.exp(sim_neg), dim=1)
        
        # Вычисляем InfoNCE Loss
        loss = -torch.log(numerator / denominator).mean()
        return loss
    
        
    def nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """
        Вычисляет NT-Xent Loss (Normalized Temperature Cross-Entropy).
        
        Args:
            z1: Латентные представления первого объекта.
            z2: Латентные представления второго объекта.
            temperature: Температурный параметр.
        
        Returns:
            torch.Tensor: Значение функции потерь.
        """
        batch_size = z1.size(0)
        
        # Косинусное подобие между всеми парами
        sim_matrix = torch.exp(torch.mm(z1, z2.t()) / temperature)
        
        # Маскирование диагональных элементов (positive pairs)
        mask = (~torch.eye(batch_size).bool()).to(z1.device)
        
        # Сумма по положительному паре
        pos_sim = torch.diag(sim_matrix)
        
        # Сумма по всем отрицательным парам
        neg_sim = sim_matrix[mask].view(batch_size, -1).sum(dim=1)
        
        loss = -torch.log(pos_sim / (pos_sim + neg_sim)).mean()
        return loss
    
    def triplet_loss(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
        """
        Вычисляет Triplet Loss.
        
        Args:
            anchor: Латентные представления anchor.
            positive: Латентные представления positive.
            negative: Латентные представления negative.
            margin: Пороговое значение для диссимиляции.
        
        Returns:
            torch.Tensor: Значение функции потерь.
        """
        distance_positive = torch.norm(anchor - positive, dim=1)
        distance_negative = torch.norm(anchor - negative, dim=1)
        
        loss = torch.relu(distance_positive - distance_negative + margin).mean()
        return loss

class Contrastive_ae_lstm(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dim: int = 128, num_layers: int = 2, dropout_rate: float = 0.2, temperature: float = 0.1):
        super(Contrastive_ae_lstm, self).__init__()
        
        # Параметры LSTM
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Энкодер с LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Линейные слои после LSTM
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, latent_dim)
        )
        
        # Параметры контрастного обучения
        self.temperature = temperature
        
        # Логирование параметров модели
        self.logger = setup_logger(f"{self.__class__.__name__}", level='debug')
        self.logger.info(f"Initializing Autoencoder with input_dim={input_dim}, latent_dim={latent_dim}")
        self.logger.log_param("input_dim", input_dim)
        self.logger.log_param("latent_dim", latent_dim)
        self.logger.log_param("hidden_dim", hidden_dim)
        self.logger.log_param("num_layers", num_layers)
        self.logger.log_param("dropout_rate", dropout_rate)
        self.logger.log_param("temperature", temperature)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через энкодер.
        
        Args:
            x: Входной тензор размерности [batch_size, sequence_length, input_dim].
        
        Returns:
            torch.Tensor: Латентные представления размерности [batch_size, latent_dim].
        """
        x = x.float()
        
        # Пропускаем данные через LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, sequence_length, hidden_dim]
        
        # Берём последнее скрытое состояние последовательности
        last_hidden_state = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # Пропускаем через линейные слои
        latent = self.fc(last_hidden_state)
        return latent
    
    def infonce_loss(self, z1: torch.Tensor, z2: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет контрастную функцию потерь (InfoNCE Loss).
        
        Args:
            z1: Латентные представления первого объекта (положительная пара).
            z2: Латентные представления второго объекта (положительная пара).
            negatives: Латентные представления отрицательных объектов.
        
        Returns:
            torch.Tensor: Значение функции потерь.
        """
        batch_size = z1.size(0)
        
        # Вычисляем косинусное подобие между всеми парами
        sim_pos = F.cosine_similarity(z1, z2, dim=-1) / self.temperature
        sim_neg = torch.matmul(z1, negatives.T) / self.temperature
        
        # Вычисляем числитель и знаменатель InfoNCE Loss
        numerator = torch.exp(sim_pos)
        denominator = numerator + torch.sum(torch.exp(sim_neg), dim=1)
        
        # Вычисляем InfoNCE Loss
        loss = -torch.log(numerator / denominator).mean()
        return loss
    
    def nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """
        Вычисляет NT-Xent Loss (Normalized Temperature Cross-Entropy).
        
        Args:
            z1: Латентные представления первого объекта.
            z2: Латентные представления второго объекта.
            temperature: Температурный параметр.
        
        Returns:
            torch.Tensor: Значение функции потерь.
        """
        batch_size = z1.size(0)
        
        # Косинусное подобие между всеми парами
        sim_matrix = torch.exp(torch.mm(z1, z2.t()) / temperature)
        
        # Маскирование диагональных элементов (positive pairs)
        mask = (~torch.eye(batch_size).bool()).to(z1.device)
        
        # Сумма по положительному паре
        pos_sim = torch.diag(sim_matrix)
        
        # Сумма по всем отрицательным парам
        neg_sim = sim_matrix[mask].view(batch_size, -1).sum(dim=1)
        
        loss = -torch.log(pos_sim / (pos_sim + neg_sim)).mean()
        return loss
    
    def triplet_loss(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
        """
        Вычисляет Triplet Loss.
        
        Args:
            anchor: Латентные представления anchor.
            positive: Латентные представления positive.
            negative: Латентные представления negative.
            margin: Пороговое значение для диссимиляции.
        
        Returns:
            torch.Tensor: Значение функции потерь.
        """
        distance_positive = torch.norm(anchor - positive, dim=1)
        distance_negative = torch.norm(anchor - negative, dim=1)
        
        loss = torch.relu(distance_positive - distance_negative + margin).mean()
        return loss