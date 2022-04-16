# A Naive Bayes model for toxic comment sentiment analysis
# CAP6640 - Spring 2022  
#   
# Portions of this code are modified from this tutorial:
# https://skimai.com/fine-tuning-bert-for-sentiment-analysis/

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from data.toxic_dataset import ToxicDataset
from sklearn.naive_bayes import MultinomialNB
from utils.roc_auc import get_auc_CV, evaluate_roc
from utils.text_processing import text_preprocessing, get_tfidf

# Load previously formatted data from disk or format data 
# for use in the naive bayes model
def preprocess(data, name):

    path = '../data/processed/clean_' + name.lower() + '.p' 

    # If the formatted data exists, load it from processed data directory
    if os.path.exists(path):

        processed_data = pickle.load(open(path, 'rb'))
        print('Loading clean_{} from disk'.format(name.lower()))

    # Else, format the data and save to processed data directory
    else:

        processed_data = []

        with tqdm(data) as tdata:

            tdata.set_description('{} Set'.format(name))

            for comment in tdata:
                processed_data.append(text_preprocessing(comment))

        # Save data to processed data directory
        pickle.dump(processed_data, open(path, 'wb'))

    return np.array(processed_data)

def run_model():
    
    # import data
    train_data = ToxicDataset(train_split=True)
    X_train, y_train = train_data.get_values()
    
    test_data = ToxicDataset(train_split=False)
    X_test, y_test = test_data.get_values()

    print('Text preprocessing')
    
    # preprocess the data
    X_train_preprocessed = preprocess(X_train, 'Train')
    X_test_preprocessed = preprocess(X_test, 'Test')

    print('Finished text preprocessing')

    print('Calculating TF-IDF')

    # Calculate TF-IDF for test and train datasets
    X_train_tfidf, X_test_tfidf = get_tfidf(X_train_preprocessed, 
                                            X_test_preprocessed)

    print('Finished TF-IDF')

    print('Calculating area under curve')
    
    auc = []
    index = np.arange(1, 10, 0.1)
    with tqdm(index) as tindex:
        for alpha in tindex:
            output = MultinomialNB(alpha=alpha)
            auc.append(get_auc_CV(output, X_train_tfidf, y_train))

    results = pd.Series(auc, index=index)

    best_alpha = np.round(results.idxmax(), 2)
    print('Best alpha: ', best_alpha)

    print('Finished AUC')
   
    print('Creating Naiave Bayes Classifier')
    # Compute predicted probabilities
    naive_bayes = MultinomialNB(alpha=best_alpha)
    naive_bayes.fit(X_train_tfidf, y_train)
    probabilities = naive_bayes.predict_proba(X_test_tfidf)

    print('Finished Naive Bayes')

    print('Evaluating ROC')
    # Evaluate the classifier
    evaluate_roc(probabilities, y_test, 'naive_bayes')

    print('Finished plotting roc')
