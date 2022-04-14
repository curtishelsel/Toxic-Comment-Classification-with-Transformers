import torch
import math
from torch import nn, Tensor
import torch.nn.functional as F
from models.positional_encoder import PositionalEncoding
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

class Transformer(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, max_len: int, dropout: float = 0.5):

        super().__init__()

        self.d_model = d_model

        self.encoder = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.decoder = nn.Linear(d_model, 2)
        
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
            
    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)

        src = self.pos_encoder(src)

        output = self.transformer_encoder(src)#, src_mask)
        output = self.decoder(output)
        output = torch.mean(output, 1)

        return output

    def generate_square_subsequent_mask(sz: int) -> Tensor:
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

