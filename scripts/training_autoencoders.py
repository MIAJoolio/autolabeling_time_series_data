from pathlib import Path
from torch.utils.data import DataLoader

from src.generation import *
from src.feature_extraction import *

def training_Basic_LAE_2l(dataset_path: str, save_path: str):
    """
    Обучает автоэнкодеры с разными размерами латентного пространства.
    
    Args:
        dataset_path (str): Путь к данным.
        save_path (str): Корневой путь для сохранения результатов.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Загрузка и нормализация данных
    dataset = Basic_dataset(data_path=dataset_path, normalize=True, norm_type="zscore")

    for idx, latent_dim in enumerate([4, 16, 32, 64, 128]):
        # Размерность входа из первого элемента датасета
        sample_x, _ = dataset[0]

        if len(sample_x.shape) == 1:
            sample_x = sample_x.unsqueeze(-1)
        input_dim = sample_x.shape[0] * sample_x.shape[1]
        
        # Создаем модель
        model = Basic_LAE_2l(input_dim=input_dim, latent_dim=latent_dim)

        # Подготовка пути сохранения для текущей модели
        new_save_path = save_path / str(latent_dim)
        new_save_path.mkdir(parents=True, exist_ok=True)

        # Разделение на train/val
        train_dt, val_dt = split_train_test(dataset=dataset, train_ratio=0.8, random_state=42)
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
            experiment_name=f"Basic_LAE_2l_latent_{latent_dim}"
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
                save_path=save_path / f'{idx}_{dt_name}_{latent_dim}_latent_space.png',
                title=f"{dt_name.capitalize()} Latent Space (dim={latent_dim})"
            )


def training_LSTM_AE(dataset_path:str, save_path:str):

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    dataset = Basic_dataset(data_path=dataset_path, normalize=True, norm_type="zscore")
    
    for inx, latent_dim in enumerate([32, 64, 128]):   
        for hidden_num in [32, 64, 128]:
            
            input_dim = dataset.series[0].shape[-1] 
            model = Basic_LSTMAE(input_dim=input_dim, hidden_dim=hidden_num, latent_dim=latent_dim, num_layers=2)
            
            new_save_path = save_path / '_'.join([str(latent_dim),str(hidden_num)])
            new_save_path.mkdir(parents=True, exist_ok=True)
            
            train_dt, val_dt = split_train_test(dataset=dataset, train_ratio=0.8, random_state=42)
            train_loader, val_loader = DataLoader(train_dt, batch_size=4, shuffle=True), DataLoader(val_dt, batch_size=4, shuffle=False)

            trained_model, history = train_autoencoder(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=100,
                lr=1e-3,
                weight_decay=1e-5,
                patience=10,
                model_save_path=new_save_path, 
                device="cuda:1",
                model_type = 'lstm'
            )    

            for dt, dt_name in zip([train_loader, val_loader], ['train', 'val']):
                latents, labels = extract_latent_features(
                model=trained_model,
                data_loader=dt,
                device="cuda:1",
                model_type='lstm')
                
                visualize_all_latent_points(latents, labels, save_path/f'{inx}_{dt_name}_{hidden_num}_{latent_dim}_latent_space.png')


def main(): 
    
    # for num_points in [100, 500]:
                
        # dt_paths = [f'data/linear_{num_points}/Linear_dataset.json', f'data/seasonal_{num_points}/Seasonal_dataset.json', f'data/ts_MNIST_{num_points}/ts_MNIST_dataset.json']
        # dt_names = [f'linear_{num_points}', f'seasonal_{num_points}', f'ts_mnist_{num_points}']

        # # dt_paths = [f'data/seasonal_{num_points}/Seasonal_dataset.json']
        # # dt_names = [f'seasonal_{num_points}']
        # dt_paths = [f'data/sas2_60/Seasonal_dataset.json']
        # dt_names = [f'data/sas2_60']
        
    #     for dt_path, dt_name in zip(dt_paths, dt_names):
    #         training_Basic_LAE_2l(dataset_path=dt_path,save_path=f'scripts/learning_results/LAE_2l/{dt_name}')
    #         # training_LSTM_AE(dataset_path=dt_path,save_path=f'scripts/learning_results/LSTM/{dt_name}')

    training_Basic_LAE_2l(f'data/synthetic/legacy/linear_100/Linear_dataset.json', 'scripts/experiment1/Linear_100/')

if __name__ == '__main__':
    main()


# def training_(dataset_path:str, save_path:str):

#     save_path = Path(save_path)
#     save_path.mkdir(parents=True, exist_ok=True)

#     dataset = Basic_dataset(data_path=dataset_path, normalize=True, norm_type="zscore")
    
#     for inx, latent_dim in enumerate([4,16,32,64,128]):    
#         input_dim = dataset.series[0].shape[-1] 
#         model = Adaptive_ae_two_layer(input_dim=input_dim, latent_dim=latent_dim)
        
#         new_save_path = save_path / str(latent_dim)
#         new_save_path.mkdir(parents=True, exist_ok=True)
        
#         train_dt, val_dt = split_train_test(dataset=dataset, train_ratio=0.8, random_state=42)
#         train_loader, val_loader = DataLoader(train_dt, batch_size=4, shuffle=True), DataLoader(val_dt, batch_size=4, shuffle=False)

#         trained_model, history = train_autoencoder(
#             model=model,
#             train_loader=train_loader,
#             val_loader=val_loader,
#             epochs=100,
#             lr=1e-3,
#             weight_decay=1e-5,
#             patience=10,
#             model_save_path=new_save_path, 
#             device="cuda:1"
#         )    

#         for dt, dt_name in zip([train_loader, val_loader], ['train', 'val']):
#             latents, labels = extract_latent_features(
#             model=trained_model,
#             data_loader=dt,
#             device="cuda:1")

#             visualize_all_latent_points(latents, labels, save_path/f'{inx}_{dt_name}_{latent_dim}_latent_space.png')

# def training_contrastive_ae(dataset_path:str, save_path:str):

#     save_path = Path(save_path)
#     save_path.mkdir(parents=True, exist_ok=True)

#     dataset = Basic_dataset(data_path=dataset_path, normalize=True, norm_type="zscore")
    
#     for inx, latent_dim in enumerate([4,16,32,64,128]):
#         for loss_function_name in ['infonce', 'nt_xent', 'triplet']:
#             input_dim = dataset.series[0].shape[-1] 
#             model = Contrastive_ae_two_layer(input_dim=input_dim, latent_dim=latent_dim)
            
#             new_save_path = save_path / loss_function_name / str(latent_dim)
#             new_save_path.mkdir(parents=True, exist_ok=True)
            
#             train_dt, val_dt = split_train_test(dataset=dataset, train_ratio=0.8, random_state=42)
#             train_loader, val_loader = DataLoader(train_dt, batch_size=4, shuffle=True), DataLoader(val_dt, batch_size=4, shuffle=False)

#             trained_model, history = train_autoencoder_contrastive(
#                 model=model,
#                 train_loader=train_loader,
#                 val_loader=val_loader,
#                 epochs=100,
#                 lr=1e-3,
#                 weight_decay=1e-5,
#                 patience=10,
#                 model_save_path=new_save_path, 
#                 device="cuda:1",
#                 loss_function=loss_function_name
#             )    

#             for dt, dt_name in zip([train_loader, val_loader], ['train', 'val']):
#                 latents, labels = extract_latent_features(
#                 model=trained_model,
#                 data_loader=dt,
#                 device="cuda:1")
                
#                 visualize_all_latent_points(latents, labels, save_path/ loss_function_name/ f'{inx}_{dt_name}_{latent_dim}_latent_space.png')

