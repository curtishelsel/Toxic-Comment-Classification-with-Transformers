import torch
import torch.nn as nn
from models.positional_encoder import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, num_tokens, dim_model, num_heads, num_enc_layers,
                    num_dec_layers, dropout):
        super().__init__()

        self.dim_model = dim_model
        
        self.positional_encoder = PositionalEncoding(dim_model=dim_model, 
                                                        dropout=dropout,
                                                        max_len=5000)

        self.embedding = nn.Embedding(num_tokens, dim_model)

        self.transformer = nn.Transformer(d_model = dim_model,
                                            nhead = num_heads, 
                                            num_encoder_layers = num_enc_layers,
                                            num_decoder_layers = num_dec_layers,
                                            dropout=dropout)

        self.fc = nn.Linear(dim_model, 1)  
            
    def forward(self, source, target, target_mask=None, source_pad_mask=None,
                    target_pad_mask=None):

        source = self.embedding(source) * math.sqrt(self.dim_model)
        target = self.embedding(target) * math.sqrt(self.dim_model)
        
        source = self.positional_encoder(source)
        target = self.positional_encoder(target)

        source = source.permute(1, 0, 2)
        target = target.permute(1, 0, 2)

        output = self.tranformer(source, target, tgt_mask=target_mask)
        output = self.fc(output)

        return output

    def get_target_mask(self, size) -> torch.tensor:
    
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0)) 

        return mask
