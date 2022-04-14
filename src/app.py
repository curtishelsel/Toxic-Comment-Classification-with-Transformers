import argparse
import models.bert as bert
import models.vanilla as vanilla 
import models.naive_bayes as naive_bayes
from models.transformer import Transformer

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', 
                        help='sets model for training \
                        and inference',
                        choices=['naive_bayes', 'bert', 'vanilla'])
    
    args = parser.parse_args()

    if args.model == 'naive_bayes':
        naive_bayes.run_model()
    elif args.model == 'bert':
        bert.run_model()
    elif args.model == 'vanilla':
        vanilla.run_model()
    else:
        print("Please provide valid model.")
