from torch.utils.data import Dataset, DataLoader
import json
import ijson
import os

class JSONDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.json_files = [f for f in os.listdir(file_path) if f.endswith('.json')]

    def __len__(self):
        return len(self.json_files)
    
    def __Jsonlen__(self):
        return sum(1 for _ in open(self.file_path))
    
    def __getitem__(self, idx):
        json_path = os.path.join(self.json_dir, self.json_files[idx])
        with open(self.json_path, 'r') as f:
            data = json.load(f)
            return data
        
    def __getJSONitem__(self, idx):
        with open(self.file_path,'r') as f:    
            for i, item in enumerate(ijson.items(f, 'item')):
                if i == idx:
                    return self.process_item(item)


    def process_item(self, item):
        return item


json_dir = '/path/to/json/files'

dataset = JSONDataset(json_dir)
dataloader = DataLoader(dataset, batch_size=10)
