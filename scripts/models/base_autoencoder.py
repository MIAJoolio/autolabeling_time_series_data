import torch
import torch.nn as nn
from .masking import TimeSeriesMasking

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, latent_dim, n_dims=1):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.n_dims = n_dims
        self.hidden_size = 512
        
        # Энкодер LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=n_dims,  # Используем n_dims напрямую
            hidden_size=self.hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Нормализация для энкодера
        self.encoder_norm = nn.LayerNorm([seq_len, self.hidden_size * 2])
        
        # Промежуточный слой
        self.latent_processor = nn.Sequential(
            nn.Linear(self.hidden_size * 2, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim, self.hidden_size * 2),
            nn.ReLU()
        )
        
        # Декодер LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=self.hidden_size * 2,
            hidden_size=self.hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Нормализация для декодера
        self.decoder_norm = nn.LayerNorm([seq_len, self.hidden_size * 2])
        
        # Выходной слой
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, n_dims)  # выход с нужным количеством признаков
        )
        self.masking = TimeSeriesMasking()

    def forward(self, x, mask=None):
        # Исправляем размерность входных данных
        if x.dim() == 2:  # [batch, seq_len]
            x = x.unsqueeze(-1)  # [batch, seq_len, 1]
        elif x.dim() == 3 and x.size(2) != self.n_dims:  # Если признаки не на последнем месте
            x = x.permute(0, 2, 1)  # Меняем местами seq_len и features
        
        # Проверяем соответствие размерности входа
        if x.size(-1) != self.n_dims:
            raise ValueError(f"Input features {x.size(-1)} != model n_dims {self.n_dims}")
        
        # Сохраняем оригинальные данные для маскирования
        original_x = x.clone()
        
        # Если есть маска, применяем её к входным данным
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)  # [batch, seq_len, 1]
            if self.n_dims > 1:
                mask = mask.expand(-1, -1, self.n_dims)  # Расширяем маску для всех признаков
            x = x * mask
        
        # Прямой проход через энкодер
        encoded, _ = self.encoder_lstm(x)
        encoded = self.encoder_norm(encoded)
        
        # Обработка латентного представления
        latent = self.latent_processor(encoded)
        
        # Декодирование
        decoded, _ = self.decoder_lstm(latent)
        decoded = self.decoder_norm(decoded)
        
        # Финальное преобразование
        reconstructed = self.output_layer(decoded)
        
        # Применяем маску к выходу
        if mask is not None:
            reconstructed = reconstructed * mask + reconstructed * (1 - mask)
        
        return reconstructed 