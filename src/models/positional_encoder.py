import math 
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout, max_len):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
    
        dimension = torch.arange(0, dim_model, 2)
        division_term = torch.exp(dimension * (-math.log(10000.0) / dim_model)) 
    
        positional_encoding = torch.zeros(max_len, 1, dim_model)
        
        positional_encoding[:, 0, 0::2] = torch.sin(position * division_term)
        positional_encoding[:, 0, 1::2] = torch.cos(position * division_term)

        self.register_buffer("positional_encoding", positional_encoding)


    def forward(self, x: torch.tensor) -> torch.tensor:

        x = x + self.positional_encoding[:x.size(0)]

        return self.dropout(x)
