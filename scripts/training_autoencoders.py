from pathlib import Path
from torch.utils.data import DataLoader

from src.generation import *
from src.feature_extraction import *


def train_default_AEs(dataset_path: str, save_path: str):
    """
    Обучает автоэнкодер для декомпозиции тренда
    
    Args:
        dataset_path (str): Путь к данным.
        save_path (str): Корневой путь для сохранения результатов.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Загрузка и нормализация данных
    dataset = Basic_dataset(data_path=dataset_path, normalize=False) # , norm_type="zscore")
    # Разделение на train/val
    train_dt, val_dt = split_train_test(dataset=dataset, train_ratio=0.8, random_state=42)
    train_loader = DataLoader(train_dt, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dt, batch_size=4, shuffle=False)
    test_loader = DataLoader(val_dt, batch_size=1, shuffle=False)

    input_dim = dataset[0][0].shape[1] # 1
    seq_len = dataset[0][0].shape[0] # 100
    input_size = input_dim * seq_len
    
    for hidden_dim in [16, 64, 256]: # [16, 32, 64, 128, 256]:
        for latent_dim in [16, 64, 256]: # [16, 32, 64, 128, 256]:

            # Собираем Linear модель
            encoder = Linear_encoder(input_size, latent_dim, hidden_dim)
            decoder = Linear_decoder(latent_dim, input_size, hidden_dim)

            lin_model = Linear_AE(encoder, decoder)
            
            # Подготовка пути сохранения для текущей модели
            new_save_path = save_path / f"Linear/{str(hidden_dim)}_{str(latent_dim)}"
            new_save_path.mkdir(parents=True, exist_ok=True)

            # Конфигурация обучения
            config = Training_config(
                epochs_num=50,
                lr=1e-3,
                weight_decay=1e-5,
                patience=10,
                model_save_path=new_save_path,
                device="cuda:1",
                experiment_name=f"Trend_Linear_AE_{str(hidden_dim)}_{str(latent_dim)}", 
                delta = 0.0
            )

            trainer = AE_trainer(
                model=lin_model,
                model_config=config,
                train_data=train_loader,
                val_data=val_loader,
                test_data=test_loader
            )

            # Обучение
            trainer.train_loop()

            # # Собираем LSTM модель
            # encoder = LSTM_encoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
            # decoder = LSTM_decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim, seq_len=seq_len)

            # lstm_model = LSTM_AE(encoder, decoder)
            
            # # Подготовка пути сохранения для текущей модели
            # new_save_path = save_path / f"LSTM/{str(hidden_dim)}_{str(latent_dim)}"
            # new_save_path.mkdir(parents=True, exist_ok=True)

            # # Конфигурация обучения
            # config = Training_config(
            #     epochs_num=50,
            #     lr=1e-3,
            #     weight_decay=1e-5,
            #     patience=10,
            #     model_save_path=new_save_path,
            #     device="cuda:1",
            #     experiment_name=f"Trend_LSTM_AE_{str(hidden_dim)}_{str(latent_dim)}",
            #     delta = 0.005
            # )

            # trainer = AE_trainer(
            #     model=lstm_model,
            #     model_config=config,
            #     train_data=train_loader,
            #     val_data=val_loader,
            #     test_data=test_loader
            # )

            # # Обучение
            # trainer.train_loop()

def check_validity(dataset_path, save_path):

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    # Загрузка и нормализация данных
    dataset = Basic_dataset(data_path=dataset_path, normalize=False) # , norm_type="zscore")
    # Разделение на train/val
    train_dt, val_dt = split_train_test(dataset=dataset, train_ratio=0.8, random_state=42)
    train_loader = DataLoader(train_dt, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dt, batch_size=4, shuffle=False)
    test_loader = DataLoader(val_dt, batch_size=1, shuffle=False)

    input_dim = dataset[0][0].shape[1] # 1
    seq_len = dataset[0][0].shape[0] # 100
    input_size = input_dim * seq_len
    
    for hidden_dim in [16, 64, 256]: # [16, 32, 64, 128, 256]:
        for latent_dim in [16, 64, 256]: # [16, 32, 64, 128, 256]:

            # Собираем Linear модель
            encoder = Linear_encoder(input_size, latent_dim, hidden_dim)
            decoder = Linear_decoder(latent_dim, input_size, hidden_dim)

            lin_model = Linear_AE(encoder, decoder)
            
            # Подготовка пути сохранения для текущей модели
            new_save_path = save_path / f"Linear/{str(hidden_dim)}_{str(latent_dim)}"
            new_save_path.mkdir(parents=True, exist_ok=True)

            # Конфигурация обучения
            config = Training_config(
                epochs_num=50,
                lr=1e-3,
                weight_decay=1e-5,
                patience=10,
                model_save_path=new_save_path,
                device="cuda:1",
                experiment_name=f"Trend_Linear_AE_{str(hidden_dim)}_{str(latent_dim)}", 
                delta = 0.0
            )

            trainer = AE_trainer(
                model=lin_model,
                model_config=config,
                train_data=train_loader,
                val_data=val_loader,
                test_data=test_loader
            )
            
            trainer.load_best_model(Path('scripts/trend_ae/linear_100/Linear')/ f"{hidden_dim}_{latent_dim}/best_model.pth")
            trainer._plot_predictions()

def main(): 

    # train_default_AEs(f'data/synthetic/trend_100/linear_100.json', 'scripts/trend_ae/linear_100/')
    
    check_validity(f'data/synthetic/trend_100_test/linear_100.json', 'scripts/trend_ae/linear_100_test/')
    # train_default_AEs(f'data/synthetic/trend_100/linear_100.json', 'scripts/trend_ae/linear_100/')


if __name__ == '__main__':
    main()
