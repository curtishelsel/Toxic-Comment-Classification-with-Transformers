# Utility functions for processing text data and calculating the 
# term frequency-inverse document frequency.
# CAP6640 - Spring 2022  
#   
# Portions of this code are modified from this tutorial:
# https://skimai.com/fine-tuning-bert-for-sentiment-analysis/

import re
import os
import pickle
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Formats a comment string for processing with the naive 
# bayes and transformer  models
def text_formatting(comment):

    # Change to lowercase
    comment = comment.lower()
    # Change 't to 'not'
    comment = re.sub(r"\'t", " not", comment)
    # Remove @name
    comment = re.sub(r'(@.*?)[\s]', ' ', comment)
    # Isolate and remove punctuations except '?'
    comment = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', comment)
    comment = re.sub(r'[^\w\s\?]', ' ', comment)
    # Remove some special characters
    comment = re.sub(r'([\;\:\|•«\n])', ' ', comment)
    # Remove stopwords except 'not' and 'can'
    comment = " ".join([word for word in comment.split()
                  if word not in stopwords.words('english')
                  or word in ['not', 'can']])
    # Remove trailing whitespace
    comment = re.sub(r'\s+', ' ', comment).strip()

    return comment
 
# Load previously formatted data from disk or format data 
# for use in the naive bayes and transformer models
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
                processed_data.append(text_formatting(comment))

        # Save data to processed data directory
        pickle.dump(processed_data, open(path, 'wb'))

    return np.array(processed_data)


# Calculates the term frequency-inverse document frequency for 
# the train and test datasets
def get_tfidf(x_train, x_test):

    tfidf = TfidfVectorizer(ngram_range=(1, 3), binary=True, smooth_idf=False)
    
    x_train_tfidf = tfidf.fit_transform(x_train)
    x_test_tfidf = tfidf.transform(x_test)

    return x_train_tfidf, x_test_tfidf

# Formats a comment string for processing with the BERT model
def bert_text_formatting(comment):

    # Remove '@name'
    comment = re.sub(r'(@.*?)[\s]', ' ', comment)

    # Replace '&amp;' with '&'
    comment = re.sub(r'&amp;', '&', comment)

    # Remove trailing whitespace
    comment = re.sub(r'\s+', ' ', comment).strip()

    return comment
