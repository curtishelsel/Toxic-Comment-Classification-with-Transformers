import torch
import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(BertClassifier, self).__init__()

        bert_output = 768
        hidden_nodes = 256
        output = 2

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Fully connected layer
        self.fully_connected = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, output)
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        last_state = outputs[0][:, 0, :]

        out = self.fully_connected(last_state)

        return out
