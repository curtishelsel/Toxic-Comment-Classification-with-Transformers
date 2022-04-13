import pandas as pd
from torch.utils.data import DataLoader

class ToxicDataset():
    def __init__(self, train_split: bool):
        train_test_path  = "train" if train_split else "test"
        data_path = f'../data/processed/processed_{train_test_path}.csv'
        data = pd.read_csv(data_path)

        self.x = data["comment_text"]
        self.y = data["target"]
        self.n_samples = len(data)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

    def get_values(self):
        return self.x.values, self.y.values
