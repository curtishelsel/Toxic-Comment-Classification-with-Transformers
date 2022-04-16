# A tranformer model for toxic comment sentiment analysis
# CAP6640 - Spring 2022  
#   
# Portions of this code are modified from these tutorials:
# https://skimai.com/fine-tuning-bert-for-sentiment-analysis/
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html

import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import utils.utils as utils
from torch.optim import Adam
import torch.nn.functional as F
from utils.roc_auc import evaluate_roc
from models.transformer import Transformer
from data.toxic_dataset import ToxicDataset
from utils.earlystopping import EarlyStopping
from torchtext.data.utils import get_tokenizer
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator
from utils.text_processing import text_formatting, preprocess

# creates the vocab matrix for a set of comments
def create_vocab(tokenizer, comments):

    special_tokens = ['<unk>', '<pad>']
    train_iterator = map(tokenizer, iter(comments))
    vocab = build_vocab_from_iterator(train_iterator, specials=special_tokens)
    vocab.set_default_index(vocab['<unk>'])

    return vocab

# Encodes the dataset of tokens values based on the vocab
def get_tokens(data, tokenizer, max_comment_length, vocab, name):

    token_ids = []

    pad_token = vocab.lookup_indices(['<pad>'])[0]

    with tqdm(data) as tdata:

        tdata.set_description('{} Set'.format(name))
        
        for comment in tdata:

            tokenized = tokenizer(comment)
            encoded = vocab.lookup_indices(tokenized)
            comment_length = len(encoded)
            difference = max_comment_length - comment_length

            # If comment is larger than max_comment_length, truncate to
            # to max_comment_length, otherwise pad to max_comment_length
            if difference < 0:
                padded_truncated_encoded = encoded[:max_comment_length]
            else:
                pad = np.full(difference, pad_token)
                padded_truncated_encoded = np.concatenate([encoded, pad])

            token_ids.append(padded_truncated_encoded)

    return np.array(token_ids, dtype='int64')

# Trains the model
def train(model, train_dataloader, device, criterion, optimizer):

    # Set model to train mode before each epoch
    model.train()

    train_loss= 0.0

    # Iterate over entire training samples (1 epoch)
    with tqdm(train_dataloader, unit='batch') as ttrain:
        for step, batch in enumerate(ttrain):

            # Shows the changes in training loss over batches
            current_loss = train_loss / (step + 1)
            description = 'Training Loss {:.4}'.format(current_loss)
            ttrain.set_description(description)
            
            token_ids, labels = batch
            
            # Push data/label to correct device
            token_ids = token_ids.to(device)
            labels = labels.to(device)

            # Reset model gradients Avoids grad accumulation
            model.zero_grad()

            # Do forward pass for current set of data
            output = model(token_ids)

            # Compute loss based on criterion
            loss = criterion(output, labels)

            # Update the running training loss
            train_loss += loss.item()

            # Computes gradient based on final loss
            loss.backward()

            # Clips the gradients to 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Computes gradient based on final loss
            optimizer.step()


# Validates the trained model on unsee data
def validate(model, val_dataloader, device, criterion):

    # Set model to eval mode before each epoch
    model.eval()

    validation_accuracy = 0.0
    validation_loss = 0.0

    # Iterate over entire validation samples (1 epoch)
    with tqdm(val_dataloader, unit='batch') as tvalidation:
        for step, batch in enumerate(tvalidation):

            # Shows the changes in training loss over batches
            current_loss = validation_loss / (step + 1)
            description = 'Validation Loss {:.4}'.format(current_loss)
            tvalidation.set_description(description)

            token_ids, labels = batch
            
            # Push data/label to correct device
            token_ids = token_ids.to(device)
            labels = labels.to(device)

            # Do forward pass for current set of data
            with torch.no_grad():
                output = model(token_ids)

            # Compute loss based on criterion
            loss = criterion(output, labels)

            # Update the running validation loss
            validation_loss += loss.item()

            predictions = torch.argmax(output, dim=1).flatten()

            # Update the running accuracy
            accuracy = (predictions == labels).cpu().numpy().mean() * 100
            validation_accuracy += accuracy

    # Compute the average loss
    validation_loss = validation_loss / len(val_dataloader)
    validation_accuracy = validation_accuracy / len(val_dataloader)

    print('Validation Accuracy: {:.1f}%'.format(validation_accuracy))

    return validation_loss

# Makes predictions on a test set
def predict(model, test_dataloader, device):


    # Set model to eval mode before each epoch
    model.eval()

    predictions = []

    # Iterate over entire set of test samples 
    with tqdm(test_dataloader, unit='batch') as ttest:
        for batch in ttest:
            ttest.set_description('Prediction')

            token_ids, labels = batch
                
            #token_ids = token_ids.type(torch.LongTensor)
            token_ids = token_ids.to(device)
            labels = labels.to(device)

            # Run through model to get predictions
            with torch.no_grad():
                output = model(token_ids)

            predictions.append(output)
    
    
    predictions = torch.cat(predictions, dim=0)

    # Get probabilities for the AUC score
    probabilities = F.softmax(predictions, dim=1).cpu().numpy()

    return probabilities

def run_model():
        
    # Training parameters
    epochs = 100
    batch_size = 64
    max_comment_length = 256
    model_path = '../models/vanilla_model.pt'

    # Model parameters
    dropout = 0.1
    embed_dim_size = 128
    num_encode_layers = 2  
    num_attention_head = 2  
    num_linear_nodes = 256

    # Clear the cuda cache and get device type
    utils.initial_setup()
    device = utils.get_device()

    # Import the model
    train_data = ToxicDataset(train_split=True)
    test_data = ToxicDataset(train_split=False)
    X, y = train_data.get_values()
    X_test, y_test = test_data.get_values()

    print('Preprocessing data')

    # Preprocess the data into token ids 
    X = preprocess(X, 'Train')
    X_test = preprocess(X_test, 'Test')
    
    print('Finished preprocessing data')

    # Set up the tokenizer and create the vocab matrix
    tokenizer = get_tokenizer('basic_english')
    vocab = create_vocab(tokenizer, X)

    print('Encoding and padding data')

    # Encode the data and pad/truncate the comment to same size
    X = get_tokens(X, tokenizer, max_comment_length, vocab, 'Train')
    X_test = get_tokens(X_test, tokenizer, max_comment_length, vocab, 'Test')
    
    print('Finished encoding and padding data')

    # Split train data into train and validation sets 
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)    

    # Create the tensors and dataloaders for training
    print('Creating tensors and dataloaders')
    train_tensors = utils.create_tensors(X_train, y_train)
    validation_tensors = utils.create_tensors(X_val, y_val)
    test_tensors = utils.create_tensors(X_test, y_test)

    train_dataloader = utils.get_dataloader(train_tensors, batch_size, train=True)
    val_dataloader = utils.get_dataloader(validation_tensors, batch_size)
    test_dataloader = utils.get_dataloader(test_tensors, batch_size)
    
    print('Finished creating tensors and dataloaders')

    print('Start Training')

    # Create model
    model = Transformer(len(vocab), embed_dim_size, 
                        num_attention_head, num_linear_nodes, 
                        num_encode_layers, max_comment_length, dropout)

    # Send model to device
    model.to(device)

    # Model utilities
    optimizer = Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=5)

    min_loss = np.inf
    # Train the model and save best model 
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}')
        train(model, train_dataloader, device, criterion, optimizer)
        loss = validate(model, val_dataloader, device, criterion)

        if loss < min_loss:
            min_loss = loss
            print('Saving best model')
            torch.save(model.state_dict(), model_path)
        
        # Stop training early if no decrease in validation loss
        early_stopping(loss)
        if early_stopping.stop:
            break

    print('Finished Training')
    
    # Load the best model
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    
    # Compute predicted probabilities on the test set
    probabilities = predict(model, test_dataloader, device)

    # Evaluate the transformer model
    evaluate_roc(probabilities, y_test, 'vanilla')
