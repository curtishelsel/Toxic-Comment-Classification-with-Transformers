import numpy as np
import pandas as pd
from tqdm import tqdm
from data.toxic_dataset import ToxicDataset
from sklearn.naive_bayes import MultinomialNB
from utils.text_processing import text_preprocessing, get_tfidf
from utils.roc_auc import get_auc_CV, evaluate_roc, plot_auc_alpha

def run_model():
    
    # import data
    train_data = ToxicDataset(train_split=True)
    X_train, y_train = train_data.get_values()
    train_length = len(X_train)
    
    test_data = ToxicDataset(train_split=False)
    X_test, y_test = test_data.get_values()
    test_length = len(X_test)

    print('Text preprocessing')
    
    X_train_preprocessed = []
    X_test_preprocessed = []

    print('Train preprocessing')
    with tqdm(X_train) as tX_train:
        for text in tX_train:
            X_train_preprocessed.append(text_preprocessing(text))

    print('Test preprocessing')
    with tqdm(X_test) as tX_test:
        for text in tX_test:
            X_test_preprocessed.append(text_preprocessing(text))

    
    X_train_preprocessed = np.array(X_train_preprocessed)
    X_test_preprocessed = np.array(X_test_preprocessed)
    print('Finished text preprocessing')

    print('Calculating TF-IDF')

    # Calculate TF-IDF
    X_train_tfidf, X_test_tfidf = get_tfidf(X_train_preprocessed, 
                                            X_test_preprocessed)

    print('Finished TF-IDF')

    print('Calculating area under curve and plotting graph')
    
    auc = []
    index = np.arange(1, 10, 0.1)
    with tqdm(index) as tindex:
        for alpha in tindex:
            value = MultinomialNB(alpha=alpha)
            auc.append(get_auc_CV(value, X_train_tfidf, y_train))

    res = pd.Series(auc, index=index)

    best_alpha = np.round(res.idxmax(), 2)
    print('Best alpha: ', best_alpha)

    
    plot_auc_alpha(res, 'naive_bayes')
    print('Finished AUC and Graph')
   
    print('Creating Naiave Bayes Classifier')
    # Compute predicted probabilities
    nb_model = MultinomialNB(alpha=best_alpha)
    nb_model.fit(X_train_tfidf, y_train)
    probs = nb_model.predict_proba(X_test_tfidf)

    print('Finished Naive Bayes')

    print('Evaluating ROC')
    # Evaluate the classifier
    evaluate_roc(probs, y_test, 'naive_bayes')

    print('Finished plotting roc')
