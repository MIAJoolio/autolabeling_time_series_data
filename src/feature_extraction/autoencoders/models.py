from typing import Literal, List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from src.utils import setup_logger, Logger 

__all__ = [
    'Basic_autoencoder',
    'Basic_LAE_2l',
    'Adaptive_LAE_2l',
    'Basic_LSTMAE'
]

class Basic_autoencoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: Any) -> Tuple[Any, Any]:
        """
        Прямой проход через автоэнкодер.
        Возвращает восстановленные данные и латентное представление.
        """
        pass

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Возвращает размерность латентного пространства."""
        pass

    def flatten_data(self, x: Any) -> Any:
        """
        Может быть переопределён в подклассах.
        По умолчанию возвращает вход без изменений.
        """
        return x
    


class Basic_LAE_2l(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32, dropout_rate: float = 0.2):
        super(Basic_LAE_2l, self).__init__()
        self._latent_dim = latent_dim

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

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @staticmethod
    def flatten_data(batch_series: torch.Tensor) -> torch.Tensor:
        """
        Преобразует данные из формы [batch_size, sequence_length, input_dim] 
        в [batch_size, sequence_length * input_dim].
        """
        batch_size, sequence_length, input_dim = batch_series.shape
        return batch_series.view(batch_size, -1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(x.shape) == 3:
            x = self.flatten_data(x)  # [4, 100, 1] → [4, 100]

        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


class Adaptive_LAE_2l(Basic_autoencoder):
    def __init__(self, input_dim: int, latent_dim: int = 32, dropout_rate: float = 0.2):
        super(Adaptive_LAE_2l, self).__init__()
        self._latent_dim = latent_dim

        hidden_dim_1 = max(input_dim // 2, latent_dim)
        hidden_dim_2 = max(input_dim // 4, latent_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_2, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_1, input_dim)
        )

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @staticmethod
    def flatten_data(batch_series: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, input_dim = batch_series.shape
        return batch_series.view(batch_size, -1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(x.shape) == 3:
            x = self.flatten_data(x)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


class Basic_LSTMAE(Basic_autoencoder):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 1,
        dropout_rate: float = 0.2
    ):
        super(Basic_LSTMAE, self).__init__()
        self._latent_dim = latent_dim

        # Encoder
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

        # Decoder
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

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, input_dim = x.shape

        encoded, (hidden, _) = self.encoder(x)
        hidden = hidden[-1]  # [batch_size, hidden_dim]
        latent = self.encoder_to_latent(hidden)

        decoder_input = self.latent_to_hidden(latent).unsqueeze(1).repeat(1, sequence_length, 1)
        reconstructed, _ = self.decoder(decoder_input)

        return reconstructed, latent