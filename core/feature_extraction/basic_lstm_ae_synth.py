import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from typing import Optional, Dict

class LSTM_Autoencoder(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        latent_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super(LSTM_Autoencoder, self).__init__()
        
        # Энкодер
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Проекция в латентное пространство
        self.latent_proj = nn.Linear(
            hidden_size * (2 if bidirectional else 1), 
            latent_size
        )
        
        # Декодер
        self.decoder_lstm = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size * (2 if bidirectional else 1),
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.decoder_out = nn.Linear(
            hidden_size * (2 if bidirectional else 1), 
            input_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Энкодер
        encoded, _ = self.encoder(x)
        
        # Латентное представление
        latent = self.latent_proj(encoded)
        
        # Декодер
        decoded, _ = self.decoder_lstm(latent)
        reconstructed = self.decoder_out(decoded)
        
        return reconstructed

def train_model(
    model,
    dataloader,
    device: torch.device,
    model_save_path: str = "models",
    model_name: str = "seasonal_ae.pth",
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    patience: int = 10,
    save_every: Optional[int] = None
) -> nn.Module:
    """
    Обучение автоэнкодера на сезонных данных с:
    - Оптимизированным сохранением моделей
    - Ранней остановкой
    - Контролем градиентов
    """
    
    # Инициализация оптимизатора и функции потерь
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=patience//2, 
        factor=0.5
    )
    
    # Подготовка директории для сохранения
    os.makedirs(model_save_path, exist_ok=True)
    
    # Перемещение модели на устройство
    model.to(device)
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch, _ in dataloader:
            batch = batch.to(device)
            if batch.dim() == 2:
                batch = batch.unsqueeze(-1)  # Добавляем размерность канала
            
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            
            # Контроль градиентов
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Логика сохранения и ранней остановки
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Сохраняем лучшую модель
            torch.save(model.state_dict(), os.path.join(model_save_path, f"best_{model_name}"))
        else:
            patience_counter += 1
        
        # Периодическое сохранение
        if save_every and (epoch+1) % save_every == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, f"epoch_{epoch+1}_{model_name}")))
        
        # Ранняя остановка
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        scheduler.step(avg_loss)
    
    # Загрузка лучшей модели
    model.load_state_dict(torch.load(os.path.join(model_save_path, f"best_{model_name}")))
    # Сохраняем финальную модель
    save_model(model, model_save_path, model_name)

    return model

def save_model(model, path="models", model_name="model.pth"):
    """Сохраняет модель в указанный путь"""
    if not os.path.exists(path):
        os.makedirs(path)
    full_path = os.path.join(path, model_name)
    torch.save(model.state_dict(), full_path)
    print(f"Model saved to {full_path}")

def load_model(model_class, model_args, path="models", model_name="model.pth"):
    """Загружает модель из указанного пути"""
    full_path = os.path.join(path, model_name)
    if os.path.exists(full_path):
        # Создаем экземпляр модели
        model = model_class(**model_args)
        # Загружаем веса
        model.load_state_dict(torch.load(full_path))
        print(f"Model loaded from {full_path}")
        return model
    print(f"No model found at {full_path}")
    return None