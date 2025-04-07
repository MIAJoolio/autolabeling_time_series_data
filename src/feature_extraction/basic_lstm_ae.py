import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Optional

class LSTM_Autoencoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        input_size: int = 1,
        hidden_size: int = 256,
        latent_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super(LSTM_Autoencoder, self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        # Энкодер
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Нормализация для энкодера
        self.encoder_norm = nn.LayerNorm([seq_len, hidden_size * (2 if bidirectional else 1)])
        
        # Обработка латентного представления
        self.latent_processor = nn.Sequential(
            nn.Linear(hidden_size * (2 if bidirectional else 1), latent_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_size, hidden_size * (2 if bidirectional else 1)),
            nn.ReLU()
        )
        
        # Декодер
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size * (2 if bidirectional else 1),
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Нормализация для декодера
        self.decoder_norm = nn.LayerNorm([seq_len, hidden_size * (2 if bidirectional else 1)])
        
        # Выходные слои
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size * (2 if bidirectional else 1), hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, input_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Проверка и корректировка размерности входных данных
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [batch, seq_len] -> [batch, seq_len, 1]
        elif x.dim() == 3 and x.size(2) != self.input_size:
            x = x.permute(0, 2, 1)  # Меняем местами seq_len и features
        
        # Энкодер
        encoded, _ = self.encoder_lstm(x)
        encoded = self.encoder_norm(encoded)
        
        # Латентное представление
        latent = self.latent_processor(encoded)
        
        # Декодер
        decoded, _ = self.decoder_lstm(latent)
        decoded = self.decoder_norm(decoded)
        
        # Выход
        reconstructed = self.output_layers(decoded)
        
        return reconstructed

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

def train_model(
    model,
    dataloader,
    criterion=None,
    optimizer=None,
    num_epochs=100,
    device="cpu",
    patience=10,
    save_best=True,
    save_every=None,  # None или число (сохранять каждые N эпох)
    model_save_path="models",
    model_save_name="trained_model.pth"
):
    """Обучение модели с оптимизированным сохранением"""
    model.to(device)
    
    if criterion is None:
        criterion = torch.nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch, _ in dataloader:
            batch = batch.to(device)
            if batch.ndim == 2:
                batch = batch.unsqueeze(-1)
            
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Логика сохранения
        if save_best and avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, model_save_path, f"best_{model_save_name}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if save_every and (epoch+1) % save_every == 0:
            save_model(model, model_save_path, f"epoch_{epoch+1}_{model_save_name}")
        
        if patience_counter >= patience:
            print("Early stopping")
            break
    
    # Сохраняем финальную модель
    save_model(model, model_save_path, model_save_name)
    return model

if __name__ == '__main__':
    None