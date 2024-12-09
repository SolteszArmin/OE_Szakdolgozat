import torch
from torch.utils.data import Dataset, DataLoader
import json
import ijson
import os

class JSONDataset(Dataset):
    def __init__(self, json_dir):
        self.json_dir = json_dir
    
    def __len__(self):
        return sum(1 for _ in open(self.json_dir))
    
    def __getitem__(self, idx):
        with open(self.json_dir,'r') as f:    
            for i, item in enumerate(ijson.items(f, 'item')):
                if i == idx:
                    return self.process_item(item)

    def process_item(self, item):
        return item


json_dir = 'C:/Users/Kasza Róbert/Downloads/A'

dataloader = DataLoader(JSONDataset('C:/Users/Kasza Róbert/Downloads/A/data_savt.json'), batch_size=10)
