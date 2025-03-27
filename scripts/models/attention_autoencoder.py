import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states):
        attention_weights = torch.softmax(self.attention(hidden_states), dim=1)
        return attention_weights * hidden_states

class AttentionAutoencoder(nn.Module):
    def __init__(self, input_size, seq_len, latent_dim, n_dims=1):
        super().__init__()
        self.seq_len = seq_len
        self.n_dims = n_dims
        self.hidden_size = 256
        
        # Энкодер
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Слой внимания
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size*2,  # *2 из-за двунаправленного LSTM
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Декодер
        self.decoder = nn.LSTM(
            input_size=self.hidden_size*2,
            hidden_size=self.hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=0.3
        )
        
        # Выходные слои
        self.output_layers = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size // 2, input_size)
        )

    def forward(self, x, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # Энкодинг
        encoded, (hidden, cell) = self.encoder(x)
        
        # Применяем внимание
        attended, _ = self.attention(encoded, encoded, encoded)
        attended = attended + encoded  # Residual connection
        
        # Декодинг
        decoder_input = attended[:, -1:, :].repeat(1, self.seq_len, 1)
        decoded, _ = self.decoder(decoder_input)
        
        # Выходной слой
        reconstructed = self.output_layers(decoded)
        
        if mask is not None:
            if self.n_dims > 1:
                mask = mask.unsqueeze(-1).repeat(1, 1, self.n_dims)
            else:
                mask = mask.unsqueeze(-1)
            reconstructed = reconstructed * mask
            x = x * mask
        
        return reconstructed 