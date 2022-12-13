import torch
import pandas as pd
import numpy as np

class RatingDataset(torch.utils.data.Dataset):
    def __init__(self, ratings_file_path):
        df = pd.read_csv(ratings_file_path, sep=' ', dtype = {'entity': int, 'item': int, 'rating': np.float32})

        self.entities = df['entity'].values
        self.items = df['item'].values
        self.ratings = df['rating'].values

        self.num_users = max(self.entities) + 1 # indices start at 0
        self.num_items = max(self.items) + 1

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, idx):
        return self.entities[idx], self.items[idx], self.ratings[idx]