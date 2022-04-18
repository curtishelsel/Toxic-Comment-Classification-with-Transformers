# A custom BERT transformer model
# CAP6640 - Spring 2022  
#   
# Portions of this code are modified from this tutorial:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html

import torch
import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p = 0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, 2)
    
    def forward(self, input_ids, attention_mask):

        out = self.bert(input_ids, attention_mask)
        out = out[1]
        out = self.dropout(out)
        out = self.linear(out)

        return out
