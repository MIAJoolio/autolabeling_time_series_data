from pathlib import Path
from torch.utils.data import DataLoader

from src.generation import *
from src.feature_extraction import *

def training_trend_LSTM_AE(dataset_path: str, save_path: str):
    """
    Обучает автоэнкодер для декомпозиции тренда
    
    Args:
        dataset_path (str): Путь к данным.
        save_path (str): Корневой путь для сохранения результатов.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Загрузка и нормализация данных
    dataset = Basic_dataset(data_path=dataset_path, normalize=True, norm_type="zscore")

    for hidden_size in [16, 32, 64]:
        for latent_dim in [16, 32, 64]:
            # Размерность входа из первого элемента датасета
            sample_x, _ = dataset[0]

            if len(sample_x.shape) == 1:
                sample_x = sample_x.unsqueeze(-1)
            input_dim = sample_x.shape[1]
            
            # Создаем модель
            model = LSTM_autoencoder(input_size=input_dim, hidden_size=hidden_size, latent_size=latent_dim)

            # Подготовка пути сохранения для текущей модели
            new_save_path = save_path / f"{str(hidden_size)}_{str(latent_dim)}"
            new_save_path.mkdir(parents=True, exist_ok=True)

            # Разделение на train/val
            train_dt, val_dt = split_train_test(dataset=dataset, train_ratio=0.6, random_state=42)
            train_loader = DataLoader(train_dt, batch_size=4, shuffle=True)
            val_loader = DataLoader(val_dt, batch_size=4, shuffle=False)

            # Конфигурация обучения
            config = Training_config(
                epochs=100,
                lr=1e-3,
                weight_decay=1e-5,
                patience=10,
                model_save_path=new_save_path,
                device="cuda:1",
                experiment_name=f"Trend_LSTM_AE_{str(hidden_size)}_{str(latent_dim)}"
            )

            # Инициализируем трейнер
            trainer = Basic_trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config
            )

            # Обучение
            trained_model, history = trainer.train()

            # Экстракция и визуализация латентных представлений
            for data_loader, dt_name in zip([train_loader, val_loader], ['train', 'val']):
                latents, labels = extract_latent_features(
                    model=trained_model,
                    data_loader=data_loader,
                    device=config.device,
                    logger=None,  # Можно передать свой логгер, если нужно
                )
                # Сохранение визуализации
                visualize_all_latent_points(
                    latents,
                    labels,
                    save_path=save_path / f'{dt_name}_{str(hidden_size)}_{str(latent_dim)}_latent_space.png',
                    title=f"{dt_name.capitalize()} Latent Space (dim={latent_dim})"
                )

def main(): 

    training_trend_LSTM_AE(f'data/synthetic/legacy/linear_100/Linear_dataset.json', 'scripts/trend_ae/linear_100/')

if __name__ == '__main__':
    main()
