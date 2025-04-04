import torch
from torch.utils.data import DataLoader
from core.feature_extraction.basic_lstm_ae import LSTM_Autoencoder, train_model
from core.generation import Synthetic_dataset
from core.utils import plot_series_grid

def main():
    # Загрузка и подготовка данных
    dataset = Synthetic_dataset(
        data_path='data/test_dataset1.json',
        normalize=True,
        norm_type='minmax'
    )
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Инициализация модели
    model = LSTM_Autoencoder(
        input_size=1,
        hidden_size=64,
        latent_size=32,
        num_layers=2,
        dropout=0.2
    )
    
    # Обучение
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    trained_model = train_model(
        model=model,
        dataloader=dataloader,
        num_epochs=100,
        device=device,
        patience=15
    )
    
    # Тестирование на примере
    test_sample, _ = dataset[0]
    test_sample = test_sample.unsqueeze(0).unsqueeze(-1).to(device)  # [1, seq_len, 1]
    
    trained_model.eval()
    with torch.no_grad():
        reconstructed = trained_model(test_sample)
    
    # Перенос на CPU для визуализации
    if device == "cuda":
        test_sample = test_sample.cpu()
        reconstructed = reconstructed.cpu()
    
    print([
            test_sample.squeeze().cpu().numpy(),
            reconstructed.squeeze().cpu().numpy()
        ])
    
    # Визуализация
    plot_series_grid(
        series_list=[
            test_sample.squeeze().cpu().numpy(),
            reconstructed.squeeze().cpu().numpy()
        ],
        labels=["Original", "Reconstructed"],
        plot_title="Autoencoder Results",
        ylabel="Value",
        xlabel="Time",
        figsize=(15, 6),
        layout='vertical'
    )

if __name__ == "__main__":
    main()