import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import logging
from transformers import BertTokenizer
from utils.roc_auc import evaluate_roc
from data.toxic_dataset import ToxicDataset
from models.bertclassifier import BertClassifier
from sklearn.model_selection import train_test_split
from utils.text_processing import bert_text_preprocessing
from torch.utils.data import TensorDataset, DataLoader

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
 
# Preprocesses the data provided into tokens and encodes them with BERT'S
# pretrained tokenizer and returns token_ids and attention masks
def preprocess(data, tokenizer, name):

    token_ids = []
    attention_masks = []

    with tqdm(data) as tdata:

        tdata.set_description('{} Text'.format(name))
        
        # For each comment in the data, encode the comment
        # with pretrained BERT tokenizer
        for comment in tdata:
            encoded = tokenizer.encode_plus(
                text=bert_text_preprocessing(comment),
                max_length=512, 
                padding='max_length',
                truncation='longest_first',
                return_attention_mask=True
                )

            token_ids.append(encoded.get('input_ids'))
            attention_masks.append(encoded.get('attention_mask'))

    return token_ids, attention_masks

# Creates the tensors for the token ids, masks, and labels
def create_tensors(token_ids, masks, label):

    token_ids = torch.tensor(token_ids)
    masks = torch.tensor(masks)
    label = torch.tensor(label)

    return token_ids, masks, label

# Creates the dataloader from the token ids, masks, and labels
def get_dataloader(tensors, batch_size, shuffle=True):

    token_ids, masks, labels = tensors
    # Create the dataset from the tensors
    data = TensorDataset(token_ids, masks, labels)
    # Create the dataloader from the dataset
    dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
    
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
            
            token_ids, masks, labels = batch
            
            # Push data/label to correct device
            token_ids = token_ids.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            # Reset model gradients Avoids grad accumulation
            model.zero_grad()

            # Do forward pass for current set of data
            output = model(token_ids, masks)

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

    return average_loss


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

            token_ids, masks, labels = batch
            
            # Push data/label to correct device
            token_ids = token_ids.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            # Do forward pass for current set of data
            with torch.no_grad():
                output = model(token_ids, masks)

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

    return val_loss, val_accuracy

# Makes predictions on a test set
def predict(model, test_dataloader, device):

    # Set model to eval mode before each epoch
    model.eval()

    predictions = []

    # Iterate over entire set of test samples 
    for batch in test_dataloader:

        token_ids, masks, labels = batch
            
        token_ids = token_ids.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        # Run through model to get predictions
        with torch.no_grad():
            output = model(token_ids, masks)

        predictions.append(output)
    
    predictions = torch.cat(predictions, dim=0)

    # Get probabilities for the AUC score
    probabilities = F.softmax(predictions, dim=1).cpu().numpy()

    return probabilities

def run_model():
        
    initial_setup()

    # Parameters for model
    epochs = 2
    batch_size = 8
    device = get_device()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    criterion = nn.CrossEntropyLoss()
    model = BertClassifier(freeze_bert=False)
    model.to(device)
    optimizer = AdamW(model.parameters())

    # Load the training and test data
    train_data = ToxicDataset(train_split=True)
    test_data = ToxicDataset(train_split=False)
    
    X, y = train_data.get_values()
    X_test, y_test = test_data.get_values()

    # Split train data into train and validation sets 
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)    
   
    # Preprocess the data into BERT token ids and masks
    print('Text preprocessing')
    train_inputs, train_masks = preprocess(X_train, tokenizer, 'Train')
    val_inputs, val_masks = preprocess(X_val, tokenizer, 'Validation')
    test_inputs, test_masks = preprocess(X_test, tokenizer, 'Test')

    print('Finished text preprocessing')

    # Create the tensors and dataloaders for training
    print('Creating tensors and dataloaders')
    train_tensors = create_tensors(train_inputs, train_masks, y_train)
    val_tensors = create_tensors(val_inputs, val_masks, y_val)
    test_tensors = create_tensors(test_inputs, test_masks, y_test)

    train_dataloader = get_dataloader(train_tensors, batch_size)
    val_dataloader = get_dataloader(val_tensors, batch_size, shuffle=False)
    test_dataloader = get_dataloader(test_tensors, batch_size, shuffle=False)
    
    print('Finished creating tensors and dataloaders')

    print('Start Training')

    train_losses = []
    val_losses= []
    accuracies = []
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}')
        train_loss = train(model, train_dataloader, device, criterion, optimizer)
        val_loss, val_accuracy = validate(model, val_dataloader, device, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(val_accuracy)
        
        print('Train Loss: {:.4f}'.format(train_loss))
        print('Validation Loss: {:.4f}'.format(val_loss))
        print('Accuracy: {:.1f}%'.format(val_accuracy))

    print('Finished Training')
    
    # Compute predicted probabilities on the test set
    probabilities = predict(model, test_dataloader, device)

    # Evaluate the Bert classifier
    evaluate_roc(probabilities, y_test, 'BERT')
