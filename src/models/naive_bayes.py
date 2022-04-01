import nltk
# Uncomment to download "stopwords"
# nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, roc_curve, auc
import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

def text_preprocessing(s,i,total):
    """
    - Lowercase the sentence
    - Change "'t" to "not"
    - Remove "@name"
    - Isolate and remove punctuations except "?"
    - Remove other special characters
    - Remove stop words except "not" and "can"
    - Remove trailing whitespace
    """
    s = s.lower()
    # Change 't to 'not'
    s = re.sub(r"\'t", " not", s)
    # Remove @name
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    # Isolate and remove punctuations except '?'
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Remove stopwords except 'not' and 'can'
    s = " ".join([word for word in s.split()
                  if word not in stopwords.words('english')
                  or word in ['not', 'can']])
    # Remove trailing whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    if (i % 1000 == 0):
        print(i, "/", total)
    return s

def get_auc_CV(model):
    """
    Return the average AUC score from cross-validation.
    """
    # Set KFold to shuffle data before the split
    kf = StratifiedKFold(5, shuffle=True, random_state=1)

    # Get AUC scores
    auc = cross_val_score(model, X_train_tfidf, y_train, scoring="roc_auc", cv=kf)

    return auc.mean()

def evaluate_roc(probs, y_true):
    """
    - Print AUC and accuracy on the test set
    - Plot ROC
    @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), 2)
    @params    y_true (np.array): an array of the true values with shape (len(y_true),)
    """
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')
       
    # Get accuracy over the test set
    y_pred = np.where(preds >= 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')
    
    # Plot ROC AUC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

if __name__ == '__main__':
    # choose device
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # import data
    train_path = data_path = f'./../../data/processed/processed_train.csv'
    test_path = data_path = f'./../../data/processed/processed_test.csv'

    train_data = pd.read_csv(train_path)
    X_train = train_data["comment_text"]
    y_train = train_data["target"]
    train_length = len(train_data)


    test_data = pd.read_csv(test_path)
    X_test = test_data["comment_text"]
    y_test = test_data["target"]
    test_length = len(test_data)

    print("finished loading data")

    # Preprocess text
    X_train_preprocessed = np.array([text_preprocessing(text,i,train_length) for i,text in enumerate(X_train)])
    print("finshed train preprocessing")
    X_val_preprocessed = np.array([text_preprocessing(text,i,test_length) for i,text in enumerate(X_test)])
    print("finshed test preprocessing")

    # Calculate TF-IDF
    tf_idf = TfidfVectorizer(ngram_range=(1, 3), binary=True, smooth_idf=False)
    X_train_tfidf = tf_idf.fit_transform(X_train_preprocessed)
    X_val_tfidf = tf_idf.transform(X_val_preprocessed)

    print("finished TF-IDF")

    res = pd.Series([get_auc_CV(MultinomialNB(alpha=i))
                for i in np.arange(1, 10, 0.1)],
                index=np.arange(1, 10, 0.1))

    best_alpha = np.round(res.idxmax(), 2)
    print('Best alpha: ', best_alpha)

    plt.plot(res)
    plt.title('AUC vs. Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('AUC')
    plt.show()

    print("finished Graph")

    # Compute predicted probabilities
    nb_model = MultinomialNB(alpha=1.8)
    nb_model.fit(X_train_tfidf, y_train)
    probs = nb_model.predict_proba(X_val_tfidf)

    print("finished Naive Bayes")

    # Evaluate the classifier
    evaluate_roc(probs, y_test)

    print("finished plotting roc")