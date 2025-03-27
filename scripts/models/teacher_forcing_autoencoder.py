import torch
import torch.nn as nn
import random
from .masking import TimeSeriesMasking
import os
import matplotlib.pyplot as plt
import numpy as np

class TeacherForcingAutoencoder(nn.Module):
    def __init__(self, seq_len, latent_dim, n_dims=1):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.n_dims = n_dims
        self.hidden_size = 128
        self.teacher_forcing_ratio = 0.5
        
        # Упрощаем энкодер
        self.encoder = nn.LSTM(
            input_size=n_dims,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=False
        )
        
        # Упрощаем декодер
        self.decoder = nn.LSTM(
            input_size=n_dims,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Изменяем выходной слой, чтобы соответствовать n_dims
        self.output_layer = nn.Linear(self.hidden_size, n_dims)
        
        # Добавляем инициализацию весов
        self.init_weights()

        self.masking = TimeSeriesMasking()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len] or [batch_size, seq_len, n_dims]
            mask: Mask tensor of shape [batch_size, seq_len] or [batch_size, seq_len, 1]
        """
        # Сохраняем оригинальный вход
        original_x = x
        
        # Убеждаемся, что размерности правильные
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [batch_size, seq_len, 1]
        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        batch_size = x.size(0)
        
        # Кодирование
        encoder_outputs, hidden = self.encoder(x)
        
        # Инициализация выхода декодера
        decoder_input = torch.zeros(batch_size, 1, self.n_dims, device=x.device)
        decoder_hidden = hidden
        outputs = []
        
        # Декодирование с teacher forcing
        for t in range(self.seq_len):
            # Получаем выход декодера
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            # Применяем выходной слой для преобразования hidden_size -> n_dims
            decoder_output = self.output_layer(decoder_output)
            outputs.append(decoder_output)
            
            # Следующий вход - это текущее значение из оригинального входа
            decoder_input = x[:, t:t+1, :]
        
        # Собираем все выходы
        reconstructed = torch.cat(outputs, dim=1)  # [batch_size, seq_len, n_dims]
        
        # Применяем маску, если она предоставлена
        if mask is not None:
            # Убеждаемся, что original_x имеет правильную размерность
            if original_x.dim() == 2:
                original_x = original_x.unsqueeze(-1)
            reconstructed = reconstructed * mask + original_x * (1 - mask)
        
        return reconstructed
