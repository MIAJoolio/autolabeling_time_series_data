from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

def infonce_loss(z1, z2, temperature=0.5):
    batch_size = z1.shape[0]
    logits = z1 @ z2.T / temperature
    labels = torch.arange(batch_size, device=z1.device)
    return F.cross_entropy(logits, labels)


def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    batch_size = z1.shape[0]
    logits_aa = z1 @ z1.T / temperature
    logits_ab = z1 @ z2.T / temperature
    logits_bb = z2 @ z2.T / temperature
    logits_ba = z2 @ z1.T / temperature

    logits_aa.fill_diagonal_(-torch.inf)
    logits_bb.fill_diagonal_(-torch.inf)
    logits_ab.fill_diagonal_(-torch.inf)
    logits_ba.fill_diagonal_(-torch.inf)

    targets_ab = torch.arange(batch_size, device=z1.device)
    targets_ba = torch.arange(batch_size, device=z1.device)

    loss_ab = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), targets_ab)
    loss_ba = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), targets_ba)

    return (loss_ab + loss_ba) / 2


def triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = (anchor - positive).pow(2).sum(1)
    distance_negative = (anchor - negative).pow(2).sum(1)
    losses = torch.relu(distance_positive - distance_negative + margin)
    return losses.mean()


class LearnableDecomposer(nn.Module):
    def __init__(
        self,
        trend_autoencoder,
        seasonal_autoencoder,
        noise_threshold=3.0
    ):
        super().__init__()
        self.trend_ae = trend_autoencoder
        self.seasonal_ae = seasonal_autoencoder
        self.noise_threshold = noise_threshold

    def forward(self, x):
        # Тренд
        trend_output = self.trend_ae(x)
        trend = trend_output['reconstructed']
        trend_latent = trend_output['latent']

        # Сезонность
        residual_after_trend = x - trend
        seasonal_output = self.seasonal_ae(residual_after_trend)
        seasonal = seasonal_output['reconstructed']
        seasonal_latent = seasonal_output['latent']

        # Шум
        noise = residual_after_trend - seasonal

        # Аномалии
        noise_abs = torch.abs(noise)
        anomaly_mask = (noise_abs > self.noise_threshold).float()

        return {
            'trend': trend,
            'seasonal': seasonal,
            'noise': noise,
            'anomaly_mask': anomaly_mask,
            'trend_latent': trend_latent,
            'seasonal_latent': seasonal_latent
        }
        

def jittering(x, sigma=0.05):
    return x + sigma * torch.randn_like(x)


def scaling(x, scale_range=(0.9, 1.1)):
    scale_factor = torch.FloatTensor(1).uniform_(*scale_range).to(x.device)
    return x * scale_factor


def masking(x, p=0.1):
    mask = torch.rand_like(x) > p
    return x * mask


def augment(x):
    x = jittering(x)
    x = scaling(x)
    x = masking(x)
    return x

class ContrastiveTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        loss_name='nt_xent',
        optimizer=None,
        device='cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.loss_name = loss_name
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=1e-3)

        self.criterion_map = {
            'infonce': self._infonce_step,
            'nt_xent': self._nt_xent_step,
            'triplet': self._triplet_step
        }

    def _get_augmented_views(self, x):
        x_aug1 = augment(x)
        x_aug2 = augment(x)
        return x_aug1, x_aug2

    def _compute_component_loss(self, component='trend', use_triplet=False, **kwargs):
        if use_triplet:
            x_neg = kwargs['x_neg']
            output_neg = self.model(x_neg)

        x_aug1, x_aug2 = kwargs['x_aug1'], kwargs['x_aug2']
        output1 = self.model(x_aug1)
        output2 = self.model(x_aug2)

        z1 = output1[f'{component}_latent']
        z2 = output2[f'{component}_latent']

        if use_triplet:
            z_neg = output_neg[f'{component}_latent']
            return triplet_loss(z1, z2, z_neg)

        else:
            return nt_xent_loss(z1, z2) if self.loss_name == 'nt_xent' else infonce_loss(z1, z2)

    def _infonce_step(self, x):
        x_aug1, x_aug2 = self._get_augmented_views(x)
        loss_trend = self._compute_component_loss('trend', x_aug1=x_aug1, x_aug2=x_aug2)
        loss_seasonal = self._compute_component_loss('seasonal', x_aug1=x_aug1, x_aug2=x_aug2)
        return loss_trend + loss_seasonal

    def _nt_xent_step(self, x):
        return self._nt_xent_step(x)

    def _triplet_step(self, x):
        x_aug1, x_aug2 = self._get_augmented_views(x)
        x_neg = torch.roll(x, shifts=1, dims=0)
        loss_trend = self._compute_component_loss('trend', use_triplet=True, x_neg=x_neg, x_aug1=x_aug1, x_aug2=x_aug2)
        loss_seasonal = self._compute_component_loss('seasonal', use_triplet=True, x_neg=x_neg, x_aug1=x_aug1, x_aug2=x_aug2)
        return loss_trend + loss_seasonal

    def train(self, epochs=100):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            for x, _ in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                x = x.to(self.device)
                criterion_func = self.criterion_map[self.loss_name]

                self.optimizer.zero_grad()
                loss = criterion_func(x)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * x.size(0)

            avg_loss = total_loss / len(self.train_loader.dataset)
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
            

class HierarchicalAttentionEncoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_levels=3, num_layers=1, dropout=0.1):
        super().__init__()
        self.num_levels = num_levels
        self.encoders = nn.ModuleList([
            nn.LSTM(input_size=input_size if i == 0 else hidden_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True) for i in range(num_levels)
        ])
        self.downsamples = nn.ModuleList([
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1) for _ in range(num_levels - 1)
        ])
        self.attention_projections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_levels)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: (B, T, F)
        :return: list of contextualized representations on each level
        """
        all_reprs = []
        for i in range(self.num_levels):
            # Кодируем уровень
            out, (h_n, _) = self.encoders[i](x)
            repr_level = h_n[-1]  # many-to-one

            # Проекция для attention
            q = self.attention_projections[i](repr_level).unsqueeze(1)  # (B, 1, H)
            k = out  # (B, T', H)
            v = out

            # Self-attention между финальным состоянием и последовательностью
            weights = torch.bmm(q, k.transpose(1, 2)) / (k.size(-1) ** 0.5)  # (B, 1, T')
            weights = F.softmax(weights, dim=-1)
            context = torch.bmm(weights, v).squeeze(1)  # (B, H)

            all_reprs.append(context)

            # Downsample для следующего уровня
            if i < self.num_levels - 1:
                x = x.transpose(1, 2)  # (B, F, T)
                x = self.downsamples[i](x).transpose(1, 2)  # (B, T', F)

        return all_reprs  # список из `num_levels` элементов: [(B, H), (B, H), ...]