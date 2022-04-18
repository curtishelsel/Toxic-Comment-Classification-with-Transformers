# A set of utility functions that are used with pytorch models
# CAP6640 - Spring 2022  
#   
# Portions of this code are modified from this tutorial:
# https://skimai.com/fine-tuning-bert-for-sentiment-analysis/

import torch
from transformers import logging
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

# Turns off warnings for BERT and clears the cuda cache
def initial_setup():
    logging.set_verbosity_error()
    torch.cuda.empty_cache()

# Checks for the available device on the system and returns device type
def get_device():

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    print('Using {} for training.'.format(device))

    return device

# Creates the tensors for the token ids, labels and, if available, masks
def create_tensors(token_ids, label, masks=None):

    token_ids = torch.tensor(token_ids)
    label = torch.tensor(label)

    if masks is None:
        return token_ids, label
    
    masks = torch.tensor(masks)

    return token_ids, masks, label

# Creates the dataloader from the token ids, labels, and if available, masks
def get_dataloader(tensors, batch_size, train=False):

    if len(tensors) == 2:
        token_ids, labels = tensors
        # Create the dataset from the tensors
        data = TensorDataset(token_ids, labels)
    else:
        token_ids, masks, labels = tensors
        # Create the dataset from the tensors
        data = TensorDataset(token_ids, masks, labels)

    # Create the dataloader from the dataset
    if train:
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)

    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    
    return dataloader
