import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import datetime
import json
import numpy as np
import os
from sklearn.decomposition import PCA

from models.base_autoencoder import LSTMAutoencoder
from models.teacher_forcing_autoencoder import TeacherForcingAutoencoder
from models.attention_autoencoder import AttentionAutoencoder
from core.generation.ts_datasets import Synthetic_dataset

def plot_reconstructions(name, originals, reconstructions, save_dir, n_samples=5):
    """
    Создает график с несколькими примерами реконструкции
    """
    plt.figure(figsize=(15, 3 * n_samples))
    indices = np.random.choice(len(originals), min(n_samples, len(originals)), replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(n_samples, 1, i + 1)
        plt.plot(originals[idx], label='Original', color='blue')
        plt.plot(reconstructions[idx], label='Reconstructed', color='red', linestyle='--')
        plt.title(f'Sample {idx}')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}_reconstructions.png"))
    plt.close()

def save_results(name, model, dataset, device, model_dir):
    """
    Сохраняет результаты реконструкции для всего датасета
    """
    os.makedirs(model_dir, exist_ok=True)
    model.eval()
    
    all_reconstructions = []
    all_originals = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            # Получаем данные из датасета
            data = dataset[i]
            if isinstance(data, (tuple, list)):
                x = data[0]
                mask = data[1] if len(data) > 1 else None
            else:
                x = data
                mask = None
            
            # Добавляем размерность батча
            x = x.unsqueeze(0).to(device)
            if mask is not None:
                mask = mask.unsqueeze(0).to(device)
            
            # Получаем реконструкцию
            reconstructed = model(x, mask)
            
            # Убираем лишние размерности
            if reconstructed.size(-1) == 1:
                reconstructed = reconstructed.squeeze(-1)
            if x.size(-1) == 1:
                x = x.squeeze(-1)
            
            # Убираем размерность батча
            reconstructed = reconstructed.squeeze(0)
            x = x.squeeze(0)
            
            all_reconstructions.append(reconstructed.cpu().numpy())
            all_originals.append(x.cpu().numpy())
    
    # Объединяем все результаты
    all_reconstructions = np.stack(all_reconstructions, axis=0)
    all_originals = np.stack(all_originals, axis=0)
    
    # Сохраняем результаты
    np.save(os.path.join(model_dir, f"{name}_reconstructions.npy"), all_reconstructions)
    np.save(os.path.join(model_dir, f"{name}_originals.npy"), all_originals)
    
    # Создаем визуализацию
    plot_reconstructions(name, all_originals, all_reconstructions, save_dir=model_dir)

def plot_all_reconstructions(name, model_dir, n_samples=10):
    """
    Создает график со всеми реконструкциями
    """
    reconstructions_path = os.path.join(model_dir, f"{name}_reconstructions.npy")
    originals_path = os.path.join(model_dir, f"{name}_originals.npy")
    
    if not os.path.exists(reconstructions_path) or not os.path.exists(originals_path):
        print(f"Файлы для {name} не найдены.")
        return
    
    reconstructions = np.load(reconstructions_path)
    originals = np.load(originals_path)
    
    num_samples = min(n_samples, len(originals))
    plt.figure(figsize=(15, 10))
    
    for i in range(num_samples):
        plt.subplot(num_samples, 1, i + 1)
        plt.plot(originals[i], label='Original', color='blue')
        plt.plot(reconstructions[i], label='Reconstructed', color='orange')
        plt.title(f'Sample {i + 1}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f"{name}_all_reconstructions.png"))
    plt.close()

def plot_latent_space(model, dataset, device, save_dir, name):
    """
    Визуализирует латентное пространство модели в 2D плоскости
    """
    model.eval()
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            # Получаем данные из датасета
            data = dataset[i]
            if isinstance(data, (tuple, list)):
                x = data[0]
                mask = data[1] if len(data) > 1 else None
            else:
                x = data
                mask = None
            
            # Добавляем размерность батча
            x = x.unsqueeze(0).to(device)
            if mask is not None:
                mask = mask.unsqueeze(0).to(device)
            
            # Получаем латентное представление
            if hasattr(model, 'get_latent_representation'):
                latent = model.get_latent_representation(x, mask)
            else:
                # Для моделей без прямого доступа к латентному представлению
                # используем выход энкодера
                if hasattr(model, 'encoder_lstm'):
                    encoded, _ = model.encoder_lstm(x)
                    latent = model.latent_processor(encoded)
                else:
                    print(f"Модель {name} не поддерживает извлечение латентного представления")
                    return
            
            # Убираем размерность батча
            latent = latent.squeeze(0)
            
            # Если латентное представление многомерное, используем PCA для уменьшения до 2D
            if latent.size(-1) > 2:
                pca = PCA(n_components=2)
                latent_2d = pca.fit_transform(latent.cpu().numpy())
            else:
                latent_2d = latent.cpu().numpy()
            
            # Сохраняем латентное представление и метку
            latent_vectors.append(latent_2d)
            labels.append(i)  # Используем индекс как метку
    
    # Объединяем все латентные представления
    latent_vectors = np.stack(latent_vectors, axis=0)
    
    # Создаем график
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], 
                         c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Sample Index')
    plt.title(f'Latent Space Visualization - {name}')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.grid(True)
    
    # Сохраняем график
    plt.savefig(os.path.join(save_dir, f"{name}_latent_space.png"))
    plt.close()
    
    # Сохраняем латентные представления
    np.save(os.path.join(save_dir, f"{name}_latent_vectors.npy"), latent_vectors)
    print(f"Латентное пространство сохранено в: {os.path.join(save_dir, f'{name}_latent_space.png')}")

def train_and_evaluate(name, model, train_loader, val_loader, device, model_dir):
    """Обучение и оценка модели"""
    os.makedirs(model_dir, exist_ok=True)
    print(f"\nРезультаты будут сохранены в директории: {model_dir}")
    
    # Создаем директорию для логов
    logs_dir = os.path.join("logs", "comparison")
    os.makedirs(logs_dir, exist_ok=True)
    print(f"Логи обучения будут сохранены в: {logs_dir}")
    
    criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True,
        min_lr=1e-6
    )
    
    n_epochs = 150
    patience = 10
    min_delta = 0.00001
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # Создаем список для хранения логов
    logs = []
    
    # Проверка входных данных в начале
    sample_batch = next(iter(train_loader))
    if isinstance(sample_batch, (tuple, list)):
        x = sample_batch[0]
    else:
        x = sample_batch
    print(f"\nTraining data check for {name}:")
    print(f"Batch shape: {x.shape}")
    print(f"Model n_dims: {model.n_dims}")
    print(f"Model seq_len: {model.seq_len}")
    print(f"Model latent_dim: {model.latent_dim}")
    
    for epoch in range(n_epochs):
        model.train()
        epoch_train_loss = 0
        num_batches = 0
        
        for x, mask in train_loader:
            x = x.to(device)
            mask = mask.to(device)
            
            # Проверяем размерности входных данных
            if x.dim() == 2:
                if model.n_dims == 1:
                    x = x.unsqueeze(-1)  # [batch, seq_len, 1]
                else:
                    # Для многомерных данных перестраиваем размерности
                    x = x.view(x.size(0), -1, model.n_dims)  # [batch, seq_len, n_dims]
            
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)  # [batch, seq_len, 1]
            
            optimizer.zero_grad()
            
            # Получаем реконструкцию
            reconstructed = model(x, mask)
            
            # Проверяем размерности выходных данных
            if reconstructed.dim() == 2:
                reconstructed = reconstructed.unsqueeze(-1)
            
            # Рассчитываем loss только на замаскированных участках
            masked_reconstructed = reconstructed * (1 - mask)
            masked_x = x * (1 - mask)
            
            # Проверяем, что есть замаскированные участки
            if torch.sum(1 - mask) > 0:
                loss = criterion(masked_reconstructed, masked_x)
                l1_loss = l1_criterion(masked_reconstructed, masked_x)
                loss = loss + 0.1 * l1_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_train_loss += loss.item()
                num_batches += 1
        
        if num_batches > 0:
            epoch_train_loss /= num_batches
        
        train_losses.append(epoch_train_loss)
        
        # Валидация
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, mask in val_loader:
                x = x.to(device)
                mask = mask.to(device)
                
                # Проверяем размерности входных данных
                if x.dim() == 2:
                    if model.n_dims == 1:
                        x = x.unsqueeze(-1)  # [batch, seq_len, 1]
                    else:
                        # Для многомерных данных перестраиваем размерности
                        x = x.view(x.size(0), -1, model.n_dims)  # [batch, seq_len, n_dims]
                
                if mask.dim() == 2:
                    mask = mask.unsqueeze(-1)  # [batch, seq_len, 1]
                
                reconstructed = model(x, mask)
                
                # Убеждаемся, что размерности совпадают
                if reconstructed.dim() == 2:
                    reconstructed = reconstructed.unsqueeze(-1)
                
                loss = criterion(reconstructed * (1 - mask), x * (1 - mask))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Получаем текущий learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Сохраняем лог эпохи
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': epoch_train_loss,
            'val_loss': val_loss,
            'learning_rate': current_lr,
            'patience_counter': patience_counter
        }
        logs.append(epoch_log)
        
        print(f'Epoch {epoch+1}/{n_epochs}:')
        print(f'Train Loss: {epoch_train_loss:.6f}')
        print(f'Val Loss: {val_loss:.6f}')
        
        # Проверка улучшения
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }
            torch.save(checkpoint, os.path.join(model_dir, f"{name}_best.pth"))
        else:
            patience_counter += 1
            print(f'Нет улучшения {patience_counter}/{patience}')
        
        if patience_counter >= patience:
            print('Раннее остановка!')
            break
        
        scheduler.step(val_loss)
    
    # Сохраняем логи в CSV
    logs_df = pd.DataFrame(logs)
    logs_df.to_csv(os.path.join(logs_dir, f"{name}_training_logs.csv"), index=False)
    
    # Сохраняем график обучения
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.savefig(os.path.join(model_dir, f"{name}_training_progress.png"))
    plt.close()
    
    # После обучения загружаем лучшую модель
    checkpoint = torch.load(os.path.join(model_dir, f"{name}_best.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.eval()
    
    # После обучения сохраняем результаты
    save_results(name, model, val_loader.dataset, device, model_dir)
    
    return best_val_loss

def compare_models(dataset_name, n_dims=1):
    print(f'Сравнение моделей на датасете {dataset_name}...')
    
    # Создаем директорию для результатов
    base_dir = os.path.join("plots", "comparison")
    os.makedirs(base_dir, exist_ok=True)
    print(f"Все результаты будут сохранены в директории: {base_dir}")
    
    # Загрузка данных
    dataset = Synthetic_dataset(f'data/{dataset_name}.json')
    
    # Получаем длину последовательности из данных
    seq_len = dataset.metadata['num_blocks'] * dataset.metadata['block_length']
    if seq_len is None:
        seq_len = len(dataset.series[0])
    
    print(f"Dataset sequence length: {seq_len}")
    
    # Преобразуем данные в тензор и создаем маску
    series_tensor = torch.tensor(dataset.series, dtype=torch.float32)
    
    # Создаем маску с правильной размерностью [batch, seq_len]
    mask = torch.ones((len(series_tensor), seq_len))
    mask[torch.rand_like(mask) < 0.5] = 0
    
    # Проверяем, что маска содержит достаточно замаскированных участков
    masked_ratio = 1 - torch.mean(mask).item()
    print(f"Masked ratio: {masked_ratio:.2f}")
    
    # Разделяем данные на train и validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(dataset)), [train_size, val_size]
    )
    
    # Создаем датасеты с масками
    train_dataset = [(series_tensor[i], mask[i]) for i in train_indices]
    val_dataset = [(series_tensor[i], mask[i]) for i in val_indices]
    
    batch_size = 32
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Создаем модели
    models = {
        'Base': LSTMAutoencoder(
            seq_len=seq_len,
            latent_dim=64,
            n_dims=n_dims
        ),
        'TeacherForcing': TeacherForcingAutoencoder(
            seq_len=seq_len,
            latent_dim=64,
            n_dims=n_dims
        ),
        # 'Attention': AttentionAutoencoder(
        #     seq_len=seq_len,
        #     latent_dim=64,
        #     n_dims=n_dims
        # )
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nТестирование модели {name}...")
        print(f"Model configuration:")
        print(f"- n_dims: {model.n_dims}")
        print(f"- seq_len: {model.seq_len}")
        print(f"- latent_dim: {model.latent_dim}")
        
        model_dir = os.path.join(base_dir, f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(model_dir, exist_ok=True)
        print(f"Результаты для модели {name} будут сохранены в: {model_dir}")
        
        model = model.to(device)
        best_loss = train_and_evaluate(name, model, train_loader, val_loader, device, model_dir)
        results[name] = best_loss
        
        # Визуализация всех реконструкций
        plot_all_reconstructions(name, model_dir)
        print(f"Графики реконструкций сохранены в: {os.path.join(model_dir, f'{name}_all_reconstructions.png')}")
        
        # Визуализация латентного пространства
        plot_latent_space(model, val_loader.dataset, device, model_dir, name)
    
    # Сохраняем результаты
    results_file = os.path.join(base_dir, 'results.txt')
    with open(results_file, 'w') as f:
        f.write('Best validation losses:\n')
        for name, loss in results.items():
            f.write(f'{name}: {loss:.4f}\n')
    print(f"\nИтоговые результаты сохранены в: {results_file}")

if __name__ == '__main__':
    compare_models('4block_dataset', n_dims=1)
    compare_models('3d_dataset', n_dims=3) 