import argparse
import models.bert as bert
import models.train_model as train_model 
import models.naive_bayes as naive_bayes
from models.transformer import Transformer

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', 
                        help='sets model for training \
                        and inference',
                        choices=['naive_bayes', 'bert', 'transformer'], 
                        default='classic')
    
    args = parser.parse_args()

    if args.model == 'naive_bayes':
        naive_bayes.run_model()
    elif args.model == 'bert':
        bert.run_model()
    elif args.model == 'transformer':
        tranformer.run_model()
    else:
        print("Please provide valid model.")
