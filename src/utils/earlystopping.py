# This file contains the class for early stopping
# during training based on validation loss
# Author: Curtis Helsel
# December 2021

import torch
import numpy as np

class EarlyStopping:

    def __init__(self, patience=15, threshold=1e-4):

        self.patience = patience
        self.epoch_count = 0
        self.best_loss = np.Inf 
        self.threshold = threshold
        self.stop = False

    # Based on the current loss of the training
    # and the patience value set, set the stop variable
    def __call__(self, current_loss):
    
        # This threshold forumla was taken from pytorch's 
        # reduce on learning rate plataeu function
        # and helps to limit how often the threshold is met
        loss_threshold = float(self.best_loss) * (1 - self.threshold)

        # If the current loss is less than threshold
        # then restart the counter and set the current
        # best loss to current loss otherwise, 
        # increment the counter 
        if current_loss < loss_threshold:
            if self.epoch_count != 0:
                print('Early stopping epoch count reset.')
            self.epoch_count = 0
            self.best_loss = current_loss
        else:
            self.epoch_count += 1
            print('Early stopping {} of {}'.format(self.epoch_count, self.patience))
    
        # If the counter is over the patience level
        # set the stop variable for early stopping
        if self.epoch_count > self.patience:
            self.stop = True
            print('Maximum epoch threshold exeeded.')
            print('Training stopping early.')


        





