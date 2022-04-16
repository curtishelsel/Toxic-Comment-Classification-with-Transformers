# Utility functions for evaluating area under the curve and the reciever
# operating characteristics curve for the models
# CAP6640 - Spring 2022  
#   
# Portions of this code are modified from this tutorial:
# https://skimai.com/fine-tuning-bert-for-sentiment-analysis/

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, auc

# Return the average AUC score from cross-validation
def get_auc_CV(model, X_train, y_train):

    # Set KFold to shuffle data before the split
    kf = StratifiedKFold(5, shuffle=True, random_state=1)

    # Get AUC scores
    auc = cross_val_score(model, X_train, y_train, scoring="roc_auc", cv=kf)

    return auc.mean()

# Evaluates the accuracy and plots the ROC curve of the model 
def evaluate_roc(probabilities, y_true, model_name):

    # Gets values from the second class and set that as the positive class
    predictions = probabilities[:, 1]
    fpr, tpr, threshold = roc_curve(y_true, predictions)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')
       
    # Get accuracy over the test set
    predictions = np.where(predictions >= 0.5, 1, 0)
    accuracy = accuracy_score(y_true, predictions)
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
    plt.savefig('../reports/figures/' + model_name + '_roc_auc.png')
