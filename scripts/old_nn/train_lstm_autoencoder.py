import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.generation.ts_datasets import Synthetic_dataset
from sklearn.preprocessing import StandardScaler
import random

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, seq_len, latent_dim, n_dims=1):
        super().__init__()
        self.seq_len = seq_len
        self.n_dims = n_dims

        # Энкодер
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=latent_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Декодер
        self.decoder = nn.LSTM(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Выходной слой
        self.output_layer = nn.Linear(latent_dim, input_size)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Энкодинг
        encoded, (hidden, cell) = self.encoder(x)
        
        # Используем последнее состояние для декодирования
        decoder_input = encoded[:, -1:, :].repeat(1, seq_len, 1)
        
        # Декодинг
        decoded, _ = self.decoder(decoder_input)
        
        # Выходной слой
        reconstructed = self.output_layer(decoded)

        # Применяем маску к выходу
        if mask is not None:
            # Расширяем маску для соответствия размерности выхода
            if self.n_dims > 1:
                # Для многомерных данных расширяем маску до нужной размерности
                mask = mask.unsqueeze(-1).repeat(1, 1, self.n_dims)
            else:
                # Для одномерных данных просто добавляем размерность
                mask = mask.unsqueeze(-1)
            reconstructed = reconstructed * mask
            x = x * mask

        return reconstructed
    
    def encode(self, x):
        """Получение латентного представления"""
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        encoded, (hidden, _) = self.encoder(x)
        return encoded[:, -1, :]

def create_mask(seq_len, batch_size, mask_ratio=0.3):
    """Создание маски с случайными пропусками"""
    # Создаем маску с правильными размерностями
    mask = torch.ones(batch_size, seq_len)
    num_masked = int(seq_len * mask_ratio)
    
    # Для каждого элемента в батче создаем свою маску
    for i in range(batch_size):
        # Создаем случайные индексы для маскирования
        masked_indices = torch.randperm(seq_len)[:num_masked]
        # Применяем маску
        mask[i, masked_indices] = 0
    
    return mask

def setup_logging(dataset_name):
    """Настройка логирования"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'training_{dataset_name}_{timestamp}.csv'
    
    # Создаем DataFrame для логов
    df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss'])
    df.to_csv(log_file, index=False)
    
    return log_file

def plot_losses(history, save_path):
    """Построение графика потерь"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_reconstruction(model, dataset, device, save_path, num_examples=5):
    """Визуализация восстановления рядов"""
    model.eval()
    with torch.no_grad():
        # Получаем случайные примеры
        indices = np.random.choice(len(dataset), num_examples, replace=False)
        plt.figure(figsize=(15, 3*num_examples))
        
        for i, idx in enumerate(indices):
            x = dataset[idx][0].unsqueeze(0).to(device)
            mask = create_mask(model.seq_len, 1).to(device)
            
            # Получаем восстановленный ряд
            reconstructed = model(x, mask)
            
            # Строим график
            plt.subplot(num_examples, 1, i+1)
            if model.n_dims > 1:
                for dim in range(model.n_dims):
                    plt.plot(x.cpu().numpy().squeeze()[:, dim], label=f'Original dim {dim}', alpha=0.7)
                    plt.plot(reconstructed.cpu().numpy().squeeze()[:, dim], label=f'Reconstructed dim {dim}', alpha=0.7)
            else:
                plt.plot(x.cpu().numpy().squeeze(), label='Original', alpha=0.7)
                plt.plot(reconstructed.cpu().numpy().squeeze(), label='Reconstructed', alpha=0.7)
            plt.plot(mask.cpu().numpy().squeeze(), label='Mask', alpha=0.3)
            plt.title(f'Example {i+1}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def custom_loss(reconstructed, target, alpha=0.2):
    # Комбинированная функция потерь
    mse_loss = nn.MSELoss()(reconstructed, target)
    
    # Добавляем L1 loss для лучшего сохранения деталей
    l1_loss = nn.L1Loss()(reconstructed, target)
    
    # Добавляем потерю на сохранение временной структуры
    temporal_loss = torch.mean(torch.abs(
        torch.diff(reconstructed, dim=1) - torch.diff(target, dim=1)
    ))
    
    return mse_loss + alpha * l1_loss + alpha * temporal_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, log_file, patience=10, min_delta=0.01):
    """Обучение модели с ранней остановкой"""
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Обучение
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch[0].to(device)
            
            # Применяем teacher forcing
            teacher_forcing_ratio = 0.5
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            
            if use_teacher_forcing:
                # Используем правильные значения на каждом шаге
                decoder_input = x
            else:
                # Используем предсказанные значения
                decoder_input = None
            
            reconstructed = model(x, decoder_input if use_teacher_forcing else None)
            loss = criterion(reconstructed, x)  # Убрали squeeze()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Валидация
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                masks = create_mask(model.seq_len, x.size(0)).to(device)
                reconstructed = model(x, masks)
                loss = criterion(reconstructed, x)  # Убрали squeeze()
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Сохранение в CSV
        with open(log_file, 'a', newline='') as f:
            f.write(f"{epoch+1},{train_loss},{val_loss}\n")
        
        # Сохранение истории
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Ранняя остановка
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    return history

def normalize_dataset(dataset):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(dataset.reshape(-1, dataset.shape[-1])).reshape(dataset.shape)
    return normalized_data, scaler

def train_dataset(dataset_name, n_dims=1):
    """Обучение модели на конкретном датасете"""
    print(f'Обучение на датасете {dataset_name}...')
    
    # Настройка логирования
    log_file = setup_logging(dataset_name)
    
    # Создание директорий
    Path('models').mkdir(exist_ok=True)
    Path('plots').mkdir(exist_ok=True)
    
    # Загрузка данных для получения метаданных
    with open(f'data/{dataset_name}.json', 'r') as f:
        data = json.load(f)
        metadata = data['metadata']
    
    # Параметры
    input_size = n_dims
    block_length = metadata['block_length']  # Длина одного блока
    num_blocks = metadata['num_blocks']  # Количество блоков
    seq_len = block_length * num_blocks  # Полная длина временного ряда
    latent_dim = 64
    batch_size = 64
    num_epochs = 200
    learning_rate = 0.001
    
    print(f'Длина последовательности: {seq_len} (блоков: {num_blocks}, длина блока: {block_length})')
    
    # Загрузка данных
    print('Загрузка данных...')
    dataset = Synthetic_dataset(f'data/{dataset_name}.json', n_dims=n_dims)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Инициализация модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Используется устройство: {device}')
    
    model = LSTMAutoencoder(input_size, seq_len, latent_dim, n_dims=n_dims).to(device)
    criterion = custom_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-6
    )
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Обучение
    print('Начало обучения...')
    history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, log_file)
    
    # Сохранение графиков
    plot_losses(history, f'plots/losses_{n_dims}d.png')
    plot_reconstruction(model, val_dataset, device, f'plots/reconstruction_{n_dims}d.png')
    
    print('Обучение завершено!')

def main():
    # Обучение на 4-блочном датасете (1D)
    train_dataset('4block_dataset', n_dims=1)
    
    # Обучение на 3D датасете
    train_dataset('3d_dataset', n_dims=3)

if __name__ == '__main__':
    main() 