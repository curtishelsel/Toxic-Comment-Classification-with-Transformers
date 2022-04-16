# A custom transformer model
# CAP6640 - Spring 2022  
#   
# Portions of this code are modified from this tutorial:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html

import torch
import math
from torch import nn, Tensor
import torch.nn.functional as F
from models.positional_encoder import PositionalEncoding
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

class Transformer(nn.Module):
    def __init__(self, num_tokens: int, embed_dim_size: int, 
                 num_attention_heads: int, num_linear_nodes: int,
                 num_encode_layers: int, max_len: int, dropout: float):

        super().__init__()

        self.embed_dim_size = embed_dim_size

        self.encoder = nn.Embedding(num_tokens, embed_dim_size)
    
        self.positional_encoder = PositionalEncoding(embed_dim_size, 
                                                        dropout, max_len)

        encoder_layers = TransformerEncoderLayer(embed_dim_size,
                                                    num_attention_heads,
                                                    num_linear_nodes, dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layers, 
                                                        num_encode_layers)

        self.decoder = nn.Linear(embed_dim_size, 2)
        
    def forward(self, src: Tensor) -> Tensor:

        src = self.encoder(src) * math.sqrt(self.embed_dim_size)
        src = self.positional_encoder(src)

        output = self.transformer_encoder(src)
        output = self.decoder(output)
        output = torch.mean(output, 1)

        return output
