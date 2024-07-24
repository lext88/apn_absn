import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class TextDataset(Dataset):
    def __init__(self, tokenizer_name, file_path, max_length=128, split="train"):
        """
        Args:
            tokenizer_name (str): Name or path of the tokenizer to use.
            file_path (str): Path to the dataset file (CSV or JSON).
            max_length (int): Maximum sequence length for padding/truncation.
            split (str): Indicates whether the dataset is for training, validation, or testing.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.split = split
        
        # Load dataset (assuming CSV or JSON format for simplicity)
        self.data = self.load_data(file_path)
    
    def load_data(self, file_path):
        import pandas as pd
        # Load dataset using pandas
        df = pd.read_csv(file_path)  # Adjust this line if using a different file format
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        return list(zip(texts, labels))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]
        
        # Tokenize text
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }


from torch.utils.data import DataLoader

def get_dataloaders(tokenizer_name, train_file, val_file, test_file, batch_size=32, max_length=128):
    # Create datasets
    train_dataset = TextDataset(tokenizer_name, train_file, max_length, split="train")
    val_dataset = TextDataset(tokenizer_name, val_file, max_length, split="val")
    test_dataset = TextDataset(tokenizer_name, test_file, max_length, split="test")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x))
    
    return train_loader, val_loader, test_loader

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
