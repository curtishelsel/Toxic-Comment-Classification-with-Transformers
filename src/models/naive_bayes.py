from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
from utils.text_processing import text_preprocessing, get_tfidf
from utils.roc_auc import get_auc_CV, evaluate_roc, plot_auc_alpha

def run_model():
    
    # import data
    train_path = data_path = f'./../data/processed/processed_train.csv'
    test_path = data_path = f'./../data/processed/processed_test.csv'

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
    X_test_preprocessed = np.array([text_preprocessing(text,i,test_length) for i,text in enumerate(X_test)])

    print("finshed train preprocessing")

    # Calculate TF-IDF
    X_train_tfidf, X_test_tfidf = get_tfidf(X_train_preprocessed, X_test_preprocessed)

    print("finished TF-IDF")

    res = pd.Series([get_auc_CV(MultinomialNB(alpha=i), X_train_tfidf, y_train)
                for i in np.arange(1, 10, 0.1)],
                index=np.arange(1, 10, 0.1))

    best_alpha = np.round(res.idxmax(), 2)
    print('Best alpha: ', best_alpha)

    plot_auc_alpha(res, 'naive_bayes')
    print("finished Graph")
   
    # Compute predicted probabilities
    nb_model = MultinomialNB(alpha=best_alpha)
    nb_model.fit(X_train_tfidf, y_train)
    probs = nb_model.predict_proba(X_test_tfidf)

    print("finished Naive Bayes")

    # Evaluate the classifier
    evaluate_roc(probs, y_test, 'naive_bayes')

    print("finished plotting roc")
