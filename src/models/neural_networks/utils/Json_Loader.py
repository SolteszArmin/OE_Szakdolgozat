from torch.utils.data import Dataset, DataLoader
import json
import os

class JSONDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.json_files = [f for f in os.listdir(file_path) if f.endswith('.json')]

    def __len__(self):
        return len(self.json_files)
    
    def __getitem__(self, idx):
        json_path = os.path.join(self.json_dir, self.json_files[idx])
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        return data
    


json_dir = '/path/to/json/files'

dataset = JSONDataset(json_dir)
dataloader = DataLoader(dataset, batch_size=10)
