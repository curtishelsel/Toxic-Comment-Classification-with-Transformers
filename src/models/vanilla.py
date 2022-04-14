import os
import torch
import pickle
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
import torch.nn.functional as F
from utils.roc_auc import evaluate_roc
from utils.earlystopping import EarlyStopping
from models.transformer import Transformer
from data.toxic_dataset import ToxicDataset
from torchtext.data.utils import get_tokenizer
from utils.text_processing import text_preprocessing
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

# Clears the cuda cache
def initial_setup():
    torch.cuda.empty_cache()

# Checks for the available device on the system and returns device type
def get_device():

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    print('Using {} for training.'.format(device))

    return device

def get_sentence_length(train, test, tokenizer):

    full_dataset = np.concatenate([train, test])

    max_len = 0
    for comment in full_dataset:
        tokenized = tokenizer(comment)
        comment_length = len(tokenized)
        if comment_length > max_len:
            max_len = comment_length
        
    return max_len

def preprocess(data, name):

    path = '../data/processed/clean_' + name.lower() + '.p' 

    if os.path.exists(path):
        processed_data = pickle.load(open(path, 'rb'))
        print('Loading clean_{} from disk'.format(name.lower()))
    else:
        processed_data = []
        with tqdm(data) as tdata:
            tdata.set_description('{} Set'.format(name))
            for sentence in tdata:
                processed_data.append(text_preprocessing(sentence))

        pickle.dump(processed_data, open(path, 'wb'))

    return processed_data

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
            pad = np.full(difference, pad_token)
            padded_encoded_token_ids = np.concatenate([encoded, pad])

            token_ids.append(padded_encoded_token_ids)

    return np.array(token_ids, dtype='int64')

# Creates the tensors for the token ids, masks, and labels
def create_tensors(token_ids, label):

    token_ids = torch.tensor(token_ids)
    label = torch.tensor(label)

    return token_ids, label
    
def get_dataloader(tensors, batch_size, train=False):

    token_ids, labels = tensors
    # Create the dataset from the tensors
    data = TensorDataset(token_ids, labels)
    if train:
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)
    # Create the dataloader from the dataset
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    
    return dataloader

# Trains the model
def train(model, train_dataloader, device, criterion, optimizer):

    # Set model to train mode before each epoch
    model.train()

    train_loss= 0.0

    # Iterate over entire training samples (1 epoch)
    with tqdm(train_dataloader, unit='batch') as tepoch:
        for step, batch in enumerate(tepoch):
            tepoch.set_description('Training')
            
            token_ids, labels = batch
            
            # Push data/label to correct device
            token_ids = token_ids.to(device)
            labels = labels.to(device)

            # Reset model gradients Avoids grad accumulation
            optimizer.zero_grad()

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

    # Compute the average loss
    average_loss = train_loss / len(train_dataloader)
        
    print('Train Loss: {:.4f}'.format(average_loss))

# Validates the trained model on unsee data
def validate(model, val_dataloader, device, criterion):

    # Set model to eval mode before each epoch
    model.eval()

    val_accuracy = 0.0
    val_loss = 0.0

    # Iterate over entire validation samples (1 epoch)
    with tqdm(val_dataloader, unit='batch') as tepoch:
        for batch in tepoch:
            tepoch.set_description('Validation')

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
            val_loss += loss.item()

            preds = torch.argmax(output, dim=1).flatten()

            # Update the running accuracy
            accuracy = (preds == labels).cpu().numpy().mean() * 100
            val_accuracy += accuracy

    # Compute the average loss
    val_loss = val_loss / len(val_dataloader)
    val_accuracy = val_accuracy / len(val_dataloader)

    print('Validation Loss: {:.4f}'.format(val_loss))
    print('Validation Accuracy: {:.1f}%'.format(val_accuracy))
    return val_loss

# Makes predictions on a test set
def predict(model, test_dataloader, device):


    # Set model to eval mode before each epoch
    model.eval()

    predictions = []

    # Iterate over entire set of test samples 
    with tqdm(test_dataloader, unit='batch') as tepoch:
        for batch in tepoch:
            tepoch.set_description('Prediction')

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
        
    initial_setup()

    device = get_device()

    # Parameters for model
    model_path = '../models/vanilla_model.pt'

    tokenizer = get_tokenizer('basic_english')

    train_data = ToxicDataset(train_split=True)
    test_data = ToxicDataset(train_split=False)
    X, y = train_data.get_values()
    X_test, y_test = test_data.get_values()

    # Preprocess the data into token ids 
    print('Preprocessing data')
    X = preprocess(X, 'Train')
    X_test = preprocess(X_test, 'Test')
    
    print('Finished preprocessing data')
    special_tokens = ['<unk>', '<pad>']
    train_iterator = map(tokenizer, iter(X))
    vocab = build_vocab_from_iterator(train_iterator, specials=special_tokens)
    vocab.set_default_index(vocab['<unk>'])
    
    max_comment_length = get_sentence_length(X, X_test, tokenizer)

    print('Encoding and padding data')
    X = get_tokens(X, tokenizer, max_comment_length, vocab, 'Train')
    X_test = get_tokens(X_test, tokenizer, max_comment_length, vocab, 'Test')
    print('Finished encoding and padding data')

    # Split train data into train and validation sets 
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)    

    # Create the tensors and dataloaders for training
    print('Creating tensors and dataloaders')
    train_tensors = create_tensors(X_train, y_train)
    val_tensors = create_tensors(X_val, y_val)
    test_tensors = create_tensors(X_test, y_test)

    train_dataloader = get_dataloader(train_tensors, batch_size, train=True)
    val_dataloader = get_dataloader(val_tensors, batch_size)
    test_dataloader = get_dataloader(test_tensors, batch_size)
    
    print('Finished creating tensors and dataloaders')

    print('Start Training')

    ntokens = len(vocab)
    nhead = 2  
    nlayers = 2  
    emsize = 128
    d_hid = 256
    dropout = 0.1
    epochs = 100
    batch_size = 64

    model = Transformer(ntokens, emsize, nhead, d_hid, 
                        nlayers, max_comment_length, dropout).to(device)

    optimizer = AdamW(model.parameters())

    criterion = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(patience=5)

    min_loss = np.inf
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}')
        train(model, train_dataloader, device, criterion, optimizer)
        loss = validate(model, val_dataloader, device, criterion)

        if loss < min_loss:
            min_loss = loss
            print('Saving best model')
            torch.save(model.state_dict(), model_path)
        
        early_stopping(loss)
        if early_stopping.stop:
            break

    print('Finished Training')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    
    # Compute predicted probabilities on the test set
    probabilities = predict(model, test_dataloader, device)

    # Evaluate the Bert classifier
    evaluate_roc(probabilities, y_test, 'vanilla')
