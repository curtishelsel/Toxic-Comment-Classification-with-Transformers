# Utility functions for processing text data and calculating the 
# term frequency-inverse document frequency.
# CAP6640 - Spring 2022  
#   
# Portions of this code are modified from this tutorial:
# https://skimai.com/fine-tuning-bert-for-sentiment-analysis/

import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Formats a comment string for processing with the naive 
# bayes and transformer  models
def text_preprocessing(comment):

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

    return s

# Calculates the term frequency-inverse document frequency for 
# the train and test datasets
def get_tfidf(x_train, x_test):

    tfidf = TfidfVectorizer(ngram_range=(1, 3), binary=True, smooth_idf=False)
    
    x_train_tfidf = tfidf.fit_transform(x_train)
    x_test_tfidf = tfidf.transform(x_test)

    return x_train_tfidf, x_test_tfidf

# Formats a comment string for processing with the BERT model
def bert_text_preprocessing(comment):

    # Remove '@name'
    comment = re.sub(r'(@.*?)[\s]', ' ', comment)

    # Replace '&amp;' with '&'
    comment = re.sub(r'&amp;', '&', comment)

    # Remove trailing whitespace
    comment = re.sub(r'\s+', ' ', comment).strip()

    return comment
