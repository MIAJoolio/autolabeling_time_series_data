import torch
import torch.nn as nn

class TimeSeriesMasking(nn.Module):
    def __init__(self, mask_value=0.0):
        super().__init__()
        self.mask_value = mask_value
    
    def forward(self, x, mask):
        """
        Применяет маску к входным данным
        Args:
            x: входные данные [batch_size, seq_len, features]
            mask: маска [batch_size, seq_len] или [batch_size, seq_len, 1]
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        return x * mask + self.mask_value * (1 - mask) 