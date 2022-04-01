import sys
sys.path.insert(0, '..')

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from transformer import Transformer
from torch.utils.data import DataLoader
from data.Dataloader import ToxicDataset


def get_device():
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    return device


def train(model, device, train_loader, optimizer, criterion):

    model.train()

    train_loss = 0.0

    with tqdm(train_loader, unit='batch') as tepoch:
        for index, batch in enumerate(tepoch):
            
            tepoch.set_description('  Training')

            input_data, target_data = batch

            input_data = input_data.to(device)
            target_data = input_data.to(device)
            
            optimizer.zero_grad()
            
            output = model(input_data)
            
            loss = criterion(output, target_data)

            loss.backward()

            optimizer.step()

            loss_difference = loss.data - train_loss

            train_loss += ((1 / (index + 1)) * loss_difference)

    print('Train Loss: {:.4f}'.format(train_loss))

    return float(train_loss)

def fit(model):
    
    epochs = 10
    batch_size = 32
    num_workers = 2
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    

    toxic_dataset = ToxicDataset(train_split=True)
    toxic_dataloader = DataLoader(dataset=toxic_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers)


    train_loss = []
    
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}')
        for batch_idx, batch_sample in enumerate(toxic_dataloader):
            data, target = batch_sample
            loss = train(model, device, train_loader, optimizer, criterion)

            break
            #train_loss.appened(loss)


if __name__ == '__main__':
    
    
    device = get_device()

    model = Transformer(num_tokens=4, dim_model=8, num_heads=2, 
            num_enc_layers=3, num_dec_layers=3, dropout=0.1)

    print(model)

    # fit(model)
    
