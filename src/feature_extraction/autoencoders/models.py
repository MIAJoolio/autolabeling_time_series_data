from typing import Literal, List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

import torch
import torch.nn as nn

from src.utils import setup_logger, Logger 

class Basic_LAE_2l(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32, dropout_rate: float = 0.2):
        super(Basic_LAE_2l, self).__init__()
        self.logger = setup_logger(f"{self.__class__.__name__}", level='debug')
        self.logger.info(f"Initializing Basic_LAE_2l with input_dim={input_dim}, latent_dim={latent_dim}")
        
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
        
        # Декодер
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, input_dim)
        )
        
        self.logger.debug("Model architecture initialized")
        self.logger.log_param("input_dim", input_dim)
        self.logger.log_param("latent_dim", latent_dim)
        self.logger.log_param("dropout_rate", dropout_rate)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.float()
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    @staticmethod
    def flatten_data(batch_series: torch.Tensor) -> torch.Tensor:
        """
        Преобразует данные из формы [batch_size, sequence_length, input_dim] в [batch_size, sequence_length * input_dim].
        Args:
            batch_series (torch.Tensor): Тензор данных формы [batch_size, sequence_length, input_dim].
        Returns:
            torch.Tensor: Тензор данных формы [batch_size, sequence_length * input_dim].
        """
        batch_size, sequence_length, input_dim = batch_series.shape
        flattened_data = batch_series.view(batch_size, -1)  # Разворачиваем последовательности
        return flattened_data


class Adaptive_LAE_2l(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32, dropout_rate: float = 0.2):
        super(Adaptive_LAE_2l, self).__init__()
        
        # Вычисляем размеры скрытых слоев динамически
        hidden_dim_1 = max(input_dim // 2, latent_dim)  
        hidden_dim_2 = max(input_dim // 4, latent_dim)  
        
        # Логирование параметров модели
        self.logger = setup_logger(f"{self.__class__.__name__}", level='debug')
        self.logger.info(f"Initializing Autoencoder with input_dim={input_dim}, latent_dim={latent_dim}")
        self.logger.debug(f"Hidden layer sizes: hidden_dim_1={hidden_dim_1}, hidden_dim_2={hidden_dim_2}")
        
        # Энкодер
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_2, latent_dim)
        )
        
        # Декодер
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_1, input_dim)
        )
        
        # Логирование архитектуры модели
        self.logger.debug("Model architecture initialized")
        self.logger.log_param("input_dim", input_dim)
        self.logger.log_param("latent_dim", latent_dim)
        self.logger.log_param("dropout_rate", dropout_rate)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.float()
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    @staticmethod
    def flatten_data(batch_series: torch.Tensor) -> torch.Tensor:
        """
        Преобразует данные из формы [batch_size, sequence_length, input_dim] в [batch_size, sequence_length * input_dim].
        Args:
            batch_series (torch.Tensor): Тензор данных формы [batch_size, sequence_length, input_dim].
        Returns:
            torch.Tensor: Тензор данных формы [batch_size, sequence_length * input_dim].
        """
        batch_size, sequence_length, input_dim = batch_series.shape
        flattened_data = batch_series.view(batch_size, -1)  # Разворачиваем последовательности
        return flattened_data


class Basic_LSTMAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 1,
        dropout_rate: float = 0.2
    ):
        """
        LSTM-автоэнкодер для временных рядов.
        
        Args:
            input_dim (int): Размерность входных данных (input_dim_per_time_step).
            hidden_dim (int): Размерность скрытого состояния LSTM.
            latent_dim (int): Размерность латентного пространства.
            num_layers (int): Количество слоев LSTM.
            dropout_rate (float): Вероятность dropout.
        """
        super(Basic_LSTMAE, self).__init__()
        
        # Логирование
        self.logger = setup_logger(f"{self.__class__.__name__}", level='debug')
        self.logger.info(
            f"Initializing LSTM Autoencoder with input_dim={input_dim}, "
            f"hidden_dim={hidden_dim}, latent_dim={latent_dim}, num_layers={num_layers}"
        )
        
        # Энкодер
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.encoder_to_latent = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        # Декодер
        self.latent_to_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=input_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Логирование параметров
        self.logger.log_param("input_dim", input_dim)
        self.logger.log_param("hidden_dim", hidden_dim)
        self.logger.log_param("latent_dim", latent_dim)
        self.logger.log_param("num_layers", num_layers)
        self.logger.log_param("dropout_rate", dropout_rate)
        self.logger.debug("Model architecture initialized")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Прямой проход через автоэнкодер.
        
        Args:
            x (torch.Tensor): Входной тензор формы [batch_size, sequence_length, input_dim].
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Восстановленный тензор и латентное представление.
        """
        batch_size, sequence_length, input_dim = x.shape
        
        # Энкодер
        encoded, (hidden, _) = self.encoder(x)  # encoded: [batch_size, sequence_length, hidden_dim]
        hidden = hidden[-1]  # Берем последнее скрытое состояние: [batch_size, hidden_dim]
        latent = self.encoder_to_latent(hidden)  # Преобразуем в латентное представление: [batch_size, latent_dim]
        
        # Декодер
        decoder_input = self.latent_to_hidden(latent).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        decoder_input = decoder_input.repeat(1, sequence_length, 1)  # Повторяем для всей последовательности
        reconstructed, _ = self.decoder(decoder_input)  # reconstructed: [batch_size, sequence_length, input_dim]
        
        return reconstructed, latent