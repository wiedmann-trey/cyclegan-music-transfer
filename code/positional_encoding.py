import torch
from torch import nn, Tensor
import math


class PositionalEncoding(nn.Module):
    # https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        division_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  
        
        pos_encoding = torch.zeros(max_len, 1, d_model)
        pos_encoding[:, 0, 0::2] = torch.sin(position * division_term)
        pos_encoding[:, 0, 1::2] = torch.cos(position * division_term)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        Residual connection + positional encoding
        """
        x = x + self.pos_encoding[:x.size(0)]
        return self.dropout(x)

